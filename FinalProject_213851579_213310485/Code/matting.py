"""Image matting (lecture: video matting - trimap, KDE, opacity map, matting).

Inputs (FAQ #7): the new background image, stabilized video and binary video.

Per frame, the lecture pipeline Trimap -> Opacity map -> Matting, with every
step computed by a course algorithm:

1. Trimap (lecture 10.1). The lecture builds the trimap from sparse scribbles
   (KDE -> P(F|c) -> geodesic D_F/D_B -> V_F/V_B). Here that classification
   is already solved: the binary mask of the background-subtraction stage is a
   dense V_F/V_B labeling and its contour is the boundary Delta. The narrow
   band B_rho(Delta) = union of balls B_rho(x), x in Delta, is computed
   exactly by eroding/dilating the mask with a disc of radius RHO:
   alpha_T = 1 inside the eroded mask, 0 outside the dilated mask, and
   phi (undecided) in the band of width ~2*RHO around the contour.

2. Color model - KDE / Parzen windows (lectures 10.1, 10.4). Foreground and
   background color densities f(c|F), f(c|B) are estimated from pixels
   adjacent to the band (the trimap-refinement step of lecture 10.2: near the
   boundary the color statistics are local and reliable). The joint 3-D KDE
   over Lab color is evaluated with the fast low-rank scheme of lecture 10.4:
   quantizing the colors into a KDE_BINS^3 histogram computes the inner
   products a_i = sum_y psi_i(y) (psi_i = bin indicators), and the Gaussian
   kernel, separable along the axes (a rank-1 factor per axis), is applied as
   three 1-D convolutions - O(|samples| + bins) instead of the quadratic
   O(|Omega| * |samples|) direct sum. Bayes with uniform priors then gives
   P(F|c) = f(c|F) / (f(c|F) + f(c|B)).

3. Opacity map (lecture 10.2). Geodesic distance maps to the confident
   regions use the lecture's weight field W(x) = grad P(F|c(x)): the discrete
   cost of a step between neighboring pixels is |P(F|c(u)) - P(F|c(v))|
   (the line integral of |<W, p'>| dl along the step, plus a small spatial
   regularizer), minimized over all paths with multi-source Dijkstra. Then

       alpha(x) = w_F(x) / (w_F(x) + w_B(x)),
       w_F(x) = D_F(x)^(-r) * P(F|c(x)),  w_B(x) = D_B(x)^(-r) * P(B|c(x)),

   with r = DIST_R = 2 (lecture: r in (0, 2]).

4. Matting (lecture 10.3). The naive blend alpha*c + (1-alpha)*B_new drags
   the old background's color into the new one, so the lecture recovers the
   pure foreground color instead: for each band pixel, over nearby confident
   foreground/background pixels (the window N(x), realized as the N_CAND
   nearest pixels of each kind),

       (x_F*, x_B*) = argmin || alpha*c(x_F) + (1-alpha)*c(x_B) - c(x) ||^2,
       J(x) = alpha(x) * c(x_F*) + (1 - alpha(x)) * B_new(x).

   On confident regions J = c (foreground) or J = B_new (background).

The alpha video stores alpha in [0, 1] scaled to [0, 255] uint8 (FAQ #5),
replicated to 3 channels.
"""
from tqdm import tqdm
import utils
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

RHO = 7             # half-width of the undecided band around the contour
SAMPLE_RADIUS = 25  # how far from the band color samples are collected
KDE_BINS = 64       # histogram bins per Lab channel (quantization step 4)
KDE_SIGMA = 2.0     # Parzen bell width, in histogram-bin units
DIST_R = 2.0        # exponent of the distance weighting, r in (0, 2]
GEO_EPS = 1e-3      # spatial regularizer of the geodesic step cost
N_CAND = 7          # confident FG/BG candidates per band pixel (window N(x))
EPS = 1e-8

# disc structuring elements realizing the balls B_rho(x) of lecture 10.1
_ERODE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RHO + 1, 2 * RHO + 1))
_SAMPLE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * SAMPLE_RADIUS + 1, 2 * SAMPLE_RADIUS + 1))
_RIM_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# Lecture 10.1 + 10.4 rule: Parzen-window (KDE) density estimate of f(c|F) /
# f(c|B), computed with the binned low-rank acceleration of lecture 10.4
# (histogram inner products a_i, then a separable Gaussian kernel applied as
# three 1-D convolutions instead of the quadratic direct sum).
def kde_3d(lab_samples):
    """Joint 3-D KDE over Lab from uint8 samples (N, 3), evaluated with the
    binned low-rank scheme of lecture 10.4. Returns a KDE_BINS^3 density."""
    # quantize colors to KDE_BINS^3 bins: hist counts = inner products a_i
    q = (lab_samples >> 2).astype(np.int32)
    idx = (q[:, 0] * KDE_BINS + q[:, 1]) * KDE_BINS + q[:, 2]
    hist = np.bincount(idx, minlength=KDE_BINS ** 3).astype(np.float32)
    # separable Gaussian kernel = rank-1 factor per axis -> three 1-D convolutions
    density = gaussian_filter(hist.reshape(KDE_BINS, KDE_BINS, KDE_BINS), KDE_SIGMA)
    density /= density.sum() + EPS
    return density + EPS


# Lecture 10.1 rule: Bayes with uniform priors over the two KDE color models,
# P(F|c) = f(c|F) / (f(c|F) + f(c|B)).
def posterior_fg(lab_roi, fg_samples, bg_samples):
    """P(F|c) per pixel: Bayes with uniform priors over the two KDE models."""
    f_f = kde_3d(fg_samples)
    f_b = kde_3d(bg_samples)
    q = lab_roi >> 2
    pf = f_f[q[..., 0], q[..., 1], q[..., 2]]
    pb = f_b[q[..., 0], q[..., 1], q[..., 2]]
    return pf / (pf + pb)


# Lecture 10.2 rule: geodesic length of a path under the weight field
# W(x) = grad P(F|c(x)); the discrete edge cost |P(u) - P(v)| is the line
# integral of |<W, p'>| dl along one pixel step (+ GEO_EPS regularizer).
def band_graph(p_fg, domain):
    """Sparse 8-neighbor graph over the True pixels of `domain` whose edge
    costs discretize the lecture's geodesic length: |<W, p'>| dl with
    W = grad P(F|c), i.e. |P(u) - P(v)| plus GEO_EPS per unit step."""
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


# Lecture 10.2 rule: D_F(x) / D_B(x) = min over paths of the geodesic length
# from x to the confident region, computed with multi-source Dijkstra.
def geodesic_dist(ids, graph, src_mask):
    """Geodesic distance d_L(x, sources) for every graph node (lecture: D_F,
    D_B), via multi-source Dijkstra."""
    return dijkstra(graph, directed=False, indices=ids[src_mask], min_only=True)


# Lecture 10.3 rule: recover the pure foreground color - search the window
# N(x) of confident pixels for the pair (x_F*, x_B*) whose alpha-blend best
# reconstructs c(x), then composite J = alpha * c(x_F*) + (1 - alpha) * B_new.
def pure_composite(frame_f, a_band, band, rim_f, rim_b, bg_roi):
    """Lecture 10.3 matting for the band pixels: find the confident pair
    (x_F*, x_B*) minimizing ||alpha*c(x_F) + (1-alpha)*c(x_B) - c(x)||^2 and
    composite with the pure foreground color c(x_F*)."""
    n = int(band.sum())
    kf, kb = min(N_CAND, int(rim_f.sum())), min(N_CAND, int(rim_b.sum()))
    a1 = a_band[:, None]
    if kf == 0 or kb == 0:
        return a1 * frame_f[band] + (1.0 - a1) * bg_roi[band]
    # the window N(x): the N_CAND nearest confident FG/BG pixels of each kind
    band_yx = np.column_stack(np.nonzero(band))
    idx_f = cKDTree(np.column_stack(np.nonzero(rim_f))).query(band_yx, k=kf)[1]
    idx_b = cKDTree(np.column_stack(np.nonzero(rim_b))).query(band_yx, k=kb)[1]
    cand_f = frame_f[rim_f][idx_f].reshape(n, kf, 3)
    cand_b = frame_f[rim_b][idx_b].reshape(n, kb, 3)
    # argmin over all (x_F, x_B) pairs of ||alpha*c(x_F)+(1-alpha)*c(x_B)-c(x)||^2
    a = a_band[:, None, None, None]
    recon = a * cand_f[:, :, None, :] + (1.0 - a) * cand_b[:, None, :, :]
    err = ((recon - frame_f[band][:, None, None, :]) ** 2).sum(-1)
    best_f = err.reshape(n, -1).argmin(1) // kb
    pure_f = cand_f[np.arange(n), best_f]
    return a1 * pure_f + (1.0 - a1) * bg_roi[band]


# One frame of the full lecture pipeline: Trimap (10.1) -> KDE color model
# (10.1/10.4) -> opacity map via geodesic distances (10.2) -> matting (10.3).
def process_frame(frame, mask, background_f):
    """Full-frame (alpha, matted) for one stabilized frame; alpha is float32
    in [0, 1], matted is float32 BGR."""
    h, w = mask.shape
    alpha = np.zeros((h, w), np.float32)
    matted = background_f.copy()
    box = utils.bbox_of_mask(mask)
    if box is None:
        return alpha, matted
    # work only on a ROI around the person (everything else is trivially BG)
    r0, c0, bh, bw = box
    m = SAMPLE_RADIUS + RHO + 5
    r0, r1 = max(0, r0 - m), min(h, r0 + bh + m)
    c0, c1 = max(0, c0 - m), min(w, c0 + bw + m)

    # Trimap (lecture 10.1): erode/dilate with a disc of radius RHO realizes
    # the narrow band B_rho(Delta) around the mask contour Delta exactly;
    # alpha_T = 1 on the eroded core, 0 outside the dilation, phi in the band
    mask_roi = mask[r0:r1, c0:c1]
    fg_core = cv2.erode(mask_roi, _ERODE_K)
    dilated = cv2.dilate(mask_roi, _ERODE_K)
    band_u8 = dilated - fg_core
    band = band_u8.astype(bool)
    fg_bool = fg_core.astype(bool)

    # confident pixels are composited directly: J = c (FG) or J = B_new (BG)
    frame_f = frame[r0:r1, c0:c1].astype(np.float32)
    bg_roi = background_f[r0:r1, c0:c1]
    alpha_roi = fg_core.astype(np.float32)
    matted_roi = bg_roi.copy()
    matted_roi[fg_bool] = frame_f[fg_bool]

    if band.any():
        # KDE color model (lectures 10.1/10.2): FG/BG samples taken only from
        # pixels adjacent to the band, where the color statistics are local
        lab_roi = cv2.cvtColor(frame[r0:r1, c0:c1], cv2.COLOR_BGR2Lab)
        near_band = cv2.dilate(band_u8, _SAMPLE_K).astype(bool)
        fg_sample_px = fg_bool & near_band
        bg_sample_px = (dilated == 0) & near_band
        if fg_sample_px.any() and bg_sample_px.any():
            p_fg = posterior_fg(lab_roi, lab_roi[fg_sample_px], lab_roi[bg_sample_px])
            # confident pixels bordering the band: Omega_F / Omega_B for the
            # distance maps and the window N(x) of the matting search
            near = cv2.dilate(band_u8, _RIM_K).astype(bool)
            rim_f = fg_bool & near
            rim_b = (dilated == 0) & near
            if rim_f.any() and rim_b.any():
                # Opacity map (lecture 10.2): geodesic D_F/D_B via Dijkstra,
                # then alpha = w_F / (w_F + w_B) with
                # w_F = D_F^(-r) * P(F|c),  w_B = D_B^(-r) * P(B|c)
                domain = band | rim_f | rim_b
                ids, graph = band_graph(p_fg, domain)
                d_f = geodesic_dist(ids, graph, rim_f)[ids[band]]
                d_b = geodesic_dist(ids, graph, rim_b)[ids[band]]
                w_f = np.power(d_f, -DIST_R) * p_fg[band]
                w_b = np.power(d_b, -DIST_R) * (1.0 - p_fg[band])
                a_band = (w_f / (w_f + w_b + EPS)).astype(np.float32)
            else:
                a_band = p_fg[band].astype(np.float32)
            alpha_roi[band] = a_band
            # Matting (lecture 10.3): composite with the pure FG color
            matted_roi[band] = pure_composite(frame_f, a_band, band,
                                               rim_f, rim_b, bg_roi)

    alpha[r0:r1, c0:c1] = alpha_roi
    matted[r0:r1, c0:c1] = matted_roi
    return alpha, matted


def matting(stabilized_video_path, binary_video_path, background_image_path,
            matted_video_path, alpha_video_path):
    """Write the alpha and matted videos (lecture 10 pipeline per frame)."""
    # open video and get params
    cap = utils.open_video(stabilized_video_path)
    params = utils.get_video_parameters(cap)
    w, h, fps = params["width"], params["height"], params["fps"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    background = cv2.imread(background_image_path)
    if background is None:
        raise ValueError(f"Could not read background image {background_image_path}")
    # output frames must match the input video size (aspect ratio need not be kept)
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

        # alpha in [0, 1] scaled to [0, 255] uint8, replicated to 3 channels
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        out_alpha.write(cv2.merge([alpha_u8, alpha_u8, alpha_u8]))
        out_matted.write(np.clip(matted, 0, 255).astype(np.uint8))

    out_alpha.release()
    out_matted.release()
    cv2.destroyAllWindows()
