"""
Video matting:

for each frame build a trimap from the mask estimate a color model (KDE)
compute alpha (opacity) using geodesic distances then blend in the new
background using the cleanest matching foreground color found nearby

alpha video stores alpha in [0, 1] scaled to [0, 255] replicated to 3 channels
"""
from tqdm import tqdm
import utils
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

RHO = 7             # half-width of the undecided band around the mask edge
SAMPLE_RADIUS = 25  # how far from the band color samples are collected
KDE_BINS = 64       # color histogram bins per Lab channel
KDE_SIGMA = 2.0     # amount of smoothing applied to the histogram
DIST_R = 2.0        # how strongly distance affects alpha
GEO_EPS = 1e-3      # small extra cost per step, avoids zero-cost paths
N_CAND = 7          # number of nearby confident pixels checked per band pixel
EPS = 1e-8          # avoids divide-by-zero

# disc-shaped kernels used for erosion/dilation
_ERODE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RHO + 1, 2 * RHO + 1))
_SAMPLE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * SAMPLE_RADIUS + 1, 2 * SAMPLE_RADIUS + 1))
_RIM_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def kde_3d(lab_samples):
    """Estimate a color density from sample colors (histogram + smoothing)."""
    # put each color into a bin
    q = (lab_samples >> 2).astype(np.int32)
    idx = (q[:, 0] * KDE_BINS + q[:, 1]) * KDE_BINS + q[:, 2]
    hist = np.bincount(idx, minlength=KDE_BINS ** 3).astype(np.float32)
    # smooth the histogram - this turns it into a KDE
    density = gaussian_filter(hist.reshape(KDE_BINS, KDE_BINS, KDE_BINS), KDE_SIGMA)
    density /= density.sum() + EPS
    return density + EPS


def posterior_fg(lab_roi, fg_samples, bg_samples):
    """Probability that each pixel's color belongs to the foreground."""
    f_f = kde_3d(fg_samples)
    f_b = kde_3d(bg_samples)
    q = lab_roi >> 2
    pf = f_f[q[..., 0], q[..., 1], q[..., 2]]
    pb = f_b[q[..., 0], q[..., 1], q[..., 2]]
    return pf / (pf + pb)


def band_graph(p_fg, domain):
    """Build a pixel graph: edge cost = how different two neighbors' colors are."""
    h, w = domain.shape
    ids = np.full((h, w), -1, np.int32)
    n = int(domain.sum())
    ids[domain] = np.arange(n, dtype=np.int32)
    us, vs, ws = [], [], []
    for dy, dx, step in ((0, 1, 1.0), (1, 0, 1.0),
                         (1, 1, 2.0 ** 0.5), (1, -1, 2.0 ** 0.5)):
        sl_u = (slice(0, h - dy), slice(max(0, -dx), w - max(0, dx)))
        sl_v = (slice(dy, h), slice(max(0, dx), w - max(0, -dx)))
        iu, iv = ids[sl_u], ids[sl_v]
        ok = (iu >= 0) & (iv >= 0)
        us.append(iu[ok])
        vs.append(iv[ok])
        ws.append(np.abs(p_fg[sl_u] - p_fg[sl_v])[ok] + GEO_EPS * step)
    graph = csr_matrix((np.concatenate(ws),
                        (np.concatenate(us), np.concatenate(vs))), shape=(n, n))
    return ids, graph


def geodesic_dist(ids, graph, src_mask):
    """Shortest graph distance from every pixel to the nearest source pixel."""
    return dijkstra(graph, directed=False, indices=ids[src_mask], min_only=True)


def pure_composite(frame_f, a_band, band, rim_f, rim_b, bg_roi):
    """Blend each band pixel using the best-matching nearby FG/BG colors."""
    n = int(band.sum())
    kf, kb = min(N_CAND, int(rim_f.sum())), min(N_CAND, int(rim_b.sum()))
    a1 = a_band[:, None]
    if kf == 0 or kb == 0:
        # not enough confident pixels nearby - blend directly
        return a1 * frame_f[band] + (1.0 - a1) * bg_roi[band]
    # find the N_CAND nearest confident FG/BG pixels for each band pixel
    band_yx = np.column_stack(np.nonzero(band))
    idx_f = cKDTree(np.column_stack(np.nonzero(rim_f))).query(band_yx, k=kf)[1]
    idx_b = cKDTree(np.column_stack(np.nonzero(rim_b))).query(band_yx, k=kb)[1]
    cand_f = frame_f[rim_f][idx_f].reshape(n, kf, 3)
    cand_b = frame_f[rim_b][idx_b].reshape(n, kb, 3)
    # try every FG/BG pair, keep the one that best reconstructs this pixel's color
    a = a_band[:, None, None, None]
    recon = a * cand_f[:, :, None, :] + (1.0 - a) * cand_b[:, None, :, :]
    err = ((recon - frame_f[band][:, None, None, :]) ** 2).sum(-1)
    best_f = err.reshape(n, -1).argmin(1) // kb
    pure_f = cand_f[np.arange(n), best_f]
    return a1 * pure_f + (1.0 - a1) * bg_roi[band]


def process_frame(frame, mask, background_f):
    """Compute alpha and the matted frame for one video frame."""
    h, w = mask.shape
    alpha = np.zeros((h, w), np.float32)
    matted = background_f.copy()
    box = utils.bbox_of_mask(mask)
    if box is None:
        return alpha, matted  # no person found in this frame
    # crop to the area around the person - the rest is trivially background
    r0, c0, bh, bw = box
    m = SAMPLE_RADIUS + RHO + 5
    r0, r1 = max(0, r0 - m), min(h, r0 + bh + m)
    c0, c1 = max(0, c0 - m), min(w, c0 + bw + m)

    # trimap: erode/dilate the mask to get a confident core, confident
    # background, and an undecided band in between
    mask_roi = mask[r0:r1, c0:c1]
    fg_core = cv2.erode(mask_roi, _ERODE_K)
    dilated = cv2.dilate(mask_roi, _ERODE_K)
    band_u8 = dilated - fg_core
    band = band_u8.astype(bool)
    fg_bool = fg_core.astype(bool)

    # confident pixels are easy: keep the frame color or the new background
    frame_f = frame[r0:r1, c0:c1].astype(np.float32)
    bg_roi = background_f[r0:r1, c0:c1]
    alpha_roi = fg_core.astype(np.float32)
    matted_roi = bg_roi.copy()
    matted_roi[fg_bool] = frame_f[fg_bool]

    if band.any():
        # collect FG/BG color samples close to the band
        lab_roi = cv2.cvtColor(frame[r0:r1, c0:c1], cv2.COLOR_BGR2Lab)
        near_band = cv2.dilate(band_u8, _SAMPLE_K).astype(bool)
        fg_sample_px = fg_bool & near_band
        bg_sample_px = (dilated == 0) & near_band
        if fg_sample_px.any() and bg_sample_px.any():
            p_fg = posterior_fg(lab_roi, lab_roi[fg_sample_px], lab_roi[bg_sample_px])
            # confident pixels right next to the band, used below
            near = cv2.dilate(band_u8, _RIM_K).astype(bool)
            rim_f = fg_bool & near
            rim_b = (dilated == 0) & near
            if rim_f.any() and rim_b.any():
                # alpha = combine geodesic distance with color probability
                domain = band | rim_f | rim_b
                ids, graph = band_graph(p_fg, domain)
                d_f = geodesic_dist(ids, graph, rim_f)[ids[band]]
                d_b = geodesic_dist(ids, graph, rim_b)[ids[band]]
                w_f = np.power(d_f, -DIST_R) * p_fg[band]
                w_b = np.power(d_b, -DIST_R) * (1.0 - p_fg[band])
                a_band = (w_f / (w_f + w_b + EPS)).astype(np.float32)
            else:
                a_band = p_fg[band].astype(np.float32)  # not enough pixels for distance
            alpha_roi[band] = a_band
            # blend using the cleanest matching foreground color nearby
            matted_roi[band] = pure_composite(frame_f, a_band, band,
                                               rim_f, rim_b, bg_roi)

    alpha[r0:r1, c0:c1] = alpha_roi
    matted[r0:r1, c0:c1] = matted_roi
    return alpha, matted


def matting(stabilized_video_path, binary_video_path, background_image_path,
            matted_video_path, alpha_video_path):
    """Process a video frame by frame and write out the alpha and matted videos."""
    # read video info
    cap = utils.open_video(stabilized_video_path)
    params = utils.get_video_parameters(cap)
    w, h, fps = params["width"], params["height"], params["fps"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    background = cv2.imread(background_image_path)
    if background is None:
        raise ValueError(f"Could not read background image {background_image_path}")
    # output frames must match the input video size
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
    background_f = background.astype(np.float32)

    # create output videos
    out_alpha = utils.create_video(alpha_video_path, fps, w, h)
    out_matted = utils.create_video(matted_video_path, fps, w, h)

    frame_pairs = zip(utils.frames_of(stabilized_video_path),
                      utils.frames_of(binary_video_path))
    for frame, binary_frame in tqdm(frame_pairs, total=frame_count,
                                    desc="Matting"):
        mask = (binary_frame[..., 0] > 127).astype(np.uint8)
        alpha, matted = process_frame(frame, mask, background_f)

        # scale alpha to 0-255 and write both videos
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        out_alpha.write(cv2.merge([alpha_u8, alpha_u8, alpha_u8]))
        out_matted.write(np.clip(matted, 0, 255).astype(np.uint8))

    out_alpha.release()
    out_matted.release()
    cv2.destroyAllWindows()
