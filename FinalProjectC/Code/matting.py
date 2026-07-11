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
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from utils import open_video, create_writer, frames_of, bbox_of_mask

RHO = 7             # half-width of the undecided band around the contour
SAMPLE_RADIUS = 25  # how far from the band color samples are collected
KDE_BINS = 64       # histogram bins per Lab channel (quantization step 4)
KDE_SIGMA = 2.0     # Parzen bell width, in histogram-bin units
DIST_R = 2.0        # exponent of the distance weighting, r in (0, 2]
GEO_EPS = 1e-3      # spatial regularizer of the geodesic step cost
N_CAND = 7          # confident FG/BG candidates per band pixel (window N(x))
EPS = 1e-8

_ERODE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RHO + 1, 2 * RHO + 1))
_SAMPLE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * SAMPLE_RADIUS + 1, 2 * SAMPLE_RADIUS + 1))
_RIM_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _kde_3d(lab_samples):
    """Joint 3-D KDE over Lab from uint8 samples (N, 3), evaluated with the
    binned low-rank scheme of lecture 10.4. Returns a KDE_BINS^3 density."""
    q = (lab_samples >> 2).astype(np.int32)
    idx = (q[:, 0] * KDE_BINS + q[:, 1]) * KDE_BINS + q[:, 2]
    hist = np.bincount(idx, minlength=KDE_BINS ** 3).astype(np.float32)
    density = gaussian_filter(hist.reshape(KDE_BINS, KDE_BINS, KDE_BINS), KDE_SIGMA)
    density /= density.sum() + EPS
    return density + EPS


def _posterior_fg(lab_roi, fg_samples, bg_samples):
    """P(F|c) per pixel: Bayes with uniform priors over the two KDE models."""
    f_f = _kde_3d(fg_samples)
    f_b = _kde_3d(bg_samples)
    q = lab_roi >> 2
    pf = f_f[q[..., 0], q[..., 1], q[..., 2]]
    pb = f_b[q[..., 0], q[..., 1], q[..., 2]]
    return pf / (pf + pb)


def _band_graph(p_fg, domain):
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


def _geodesic_dist(ids, graph, src_mask):
    """Geodesic distance d_L(x, sources) for every graph node (lecture: D_F,
    D_B), via multi-source Dijkstra."""
    return dijkstra(graph, directed=False, indices=ids[src_mask], min_only=True)


def _pure_composite(frame_f, a_band, band, rim_f, rim_b, bg_roi):
    """Lecture 10.3 matting for the band pixels: find the confident pair
    (x_F*, x_B*) minimizing ||alpha*c(x_F) + (1-alpha)*c(x_B) - c(x)||^2 and
    composite with the pure foreground color c(x_F*)."""
    n = int(band.sum())
    kf, kb = min(N_CAND, int(rim_f.sum())), min(N_CAND, int(rim_b.sum()))
    a1 = a_band[:, None]
    if kf == 0 or kb == 0:
        return a1 * frame_f[band] + (1.0 - a1) * bg_roi[band]
    band_yx = np.column_stack(np.nonzero(band))
    idx_f = cKDTree(np.column_stack(np.nonzero(rim_f))).query(band_yx, k=kf)[1]
    idx_b = cKDTree(np.column_stack(np.nonzero(rim_b))).query(band_yx, k=kb)[1]
    cand_f = frame_f[rim_f][idx_f].reshape(n, kf, 3)
    cand_b = frame_f[rim_b][idx_b].reshape(n, kb, 3)
    a = a_band[:, None, None, None]
    recon = a * cand_f[:, :, None, :] + (1.0 - a) * cand_b[:, None, :, :]
    err = ((recon - frame_f[band][:, None, None, :]) ** 2).sum(-1)
    best_f = err.reshape(n, -1).argmin(1) // kb
    pure_f = cand_f[np.arange(n), best_f]
    return a1 * pure_f + (1.0 - a1) * bg_roi[band]


def _process_frame(frame, mask, background_f):
    """Full-frame (alpha, matted) for one stabilized frame; alpha is float32
    in [0, 1], matted is float32 BGR."""
    h, w = mask.shape
    alpha = np.zeros((h, w), np.float32)
    matted = background_f.copy()
    box = bbox_of_mask(mask)
    if box is None:
        return alpha, matted
    r0, c0, bh, bw = box
    m = SAMPLE_RADIUS + RHO + 5
    r0, r1 = max(0, r0 - m), min(h, r0 + bh + m)
    c0, c1 = max(0, c0 - m), min(w, c0 + bw + m)

    mask_roi = mask[r0:r1, c0:c1]
    fg_core = cv2.erode(mask_roi, _ERODE_K)
    dilated = cv2.dilate(mask_roi, _ERODE_K)
    band_u8 = dilated - fg_core
    band = band_u8.astype(bool)
    fg_bool = fg_core.astype(bool)

    frame_f = frame[r0:r1, c0:c1].astype(np.float32)
    bg_roi = background_f[r0:r1, c0:c1]
    alpha_roi = fg_core.astype(np.float32)
    matted_roi = bg_roi.copy()
    matted_roi[fg_bool] = frame_f[fg_bool]

    if band.any():
        lab_roi = cv2.cvtColor(frame[r0:r1, c0:c1], cv2.COLOR_BGR2Lab)
        near_band = cv2.dilate(band_u8, _SAMPLE_K).astype(bool)
        fg_sample_px = fg_bool & near_band
        bg_sample_px = (dilated == 0) & near_band
        if fg_sample_px.any() and bg_sample_px.any():
            p_fg = _posterior_fg(lab_roi, lab_roi[fg_sample_px], lab_roi[bg_sample_px])
            # confident pixels bordering the band: Omega_F / Omega_B for the
            # distance maps and the window N(x) of the matting search
            near = cv2.dilate(band_u8, _RIM_K).astype(bool)
            rim_f = fg_bool & near
            rim_b = (dilated == 0) & near
            if rim_f.any() and rim_b.any():
                domain = band | rim_f | rim_b
                ids, graph = _band_graph(p_fg, domain)
                d_f = _geodesic_dist(ids, graph, rim_f)[ids[band]]
                d_b = _geodesic_dist(ids, graph, rim_b)[ids[band]]
                w_f = np.power(d_f, -DIST_R) * p_fg[band]
                w_b = np.power(d_b, -DIST_R) * (1.0 - p_fg[band])
                a_band = (w_f / (w_f + w_b + EPS)).astype(np.float32)
            else:
                a_band = p_fg[band].astype(np.float32)
            alpha_roi[band] = a_band
            matted_roi[band] = _pure_composite(frame_f, a_band, band,
                                               rim_f, rim_b, bg_roi)

    alpha[r0:r1, c0:c1] = alpha_roi
    matted[r0:r1, c0:c1] = matted_roi
    return alpha, matted


def matting(stabilized_path, binary_path, background_image_path,
            matted_path, alpha_path):
    """Write alpha and matted videos. Returns (alpha_done_time,
    matted_done_time) wall-clock timestamps."""
    cap, prm = open_video(stabilized_path)
    w, h, fps = prm['width'], prm['height'], prm['fps']
    cap.release()

    background = cv2.imread(background_image_path)
    if background is None:
        raise IOError('Could not read background image: {}'.format(background_image_path))
    # output frames must match the input video size (aspect ratio need not be kept)
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
    background_f = background.astype(np.float32)

    alpha_writer = create_writer(alpha_path, fps, w, h)
    matted_writer = create_writer(matted_path, fps, w, h)

    n_done = 0
    for frame, binary_frame in zip(frames_of(stabilized_path), frames_of(binary_path)):
        mask = (binary_frame[..., 0] > 127).astype(np.uint8)
        alpha, matted = _process_frame(frame, mask, background_f)

        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        alpha_writer.write(cv2.merge([alpha_u8, alpha_u8, alpha_u8]))
        matted_writer.write(np.clip(matted, 0, 255).astype(np.uint8))

        n_done += 1
        if n_done % 50 == 0:
            print('  matting: {} frames done'.format(n_done))

    alpha_writer.release()
    alpha_done_time = time.time()
    matted_writer.release()
    matted_done_time = time.time()
    print('  matting: finished ({} frames)'.format(n_done))
    return alpha_done_time, matted_done_time
