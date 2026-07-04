"""Image matting (lecture: video matting - trimap, KDE, opacity map).

Inputs (FAQ #7): the new background image, stabilize video and binary video.

Method (per frame), following the lecture pipeline Trimap -> Opacity map ->
Matting:

1. Trimap. The binary mask from the background-subtraction stage plays the
   role of the foreground/background scribbles. Eroding it by RHO gives the
   "confident foreground" region (alpha=1), dilating it by RHO gives the
   outer edge of the narrow band; everything outside is "confident
   background" (alpha=0). The band of width ~2*RHO around the person's
   contour is the undecided region (alpha=phi) that matting must resolve.

2. Color models with KDE (Parzen windows). Foreground and background color
   distributions f(c|F), f(c|B) are estimated from the pixels adjacent to the
   band (the lecture's trimap-refinement step: near the boundary the color
   statistics are local and reliable). The KDE is computed efficiently as a
   256-bin histogram per Lab channel smoothed with a Gaussian kernel - i.e.
   a "bell" is placed around each sample - and stored as a lookup table.
   Channels are combined as independent (naive-Bayes) likelihoods.
   With uniform priors, Bayes gives  P(F|c) = f(c|F) / (f(c|F) + f(c|B)).

3. Opacity map. As in the lecture, each undecided pixel combines the color
   posterior with distance maps to the confident regions:

       alpha(x) = wF(x) / (wF(x) + wB(x)),
       wF(x) = DF(x)^(-r) * P(F|c(x)),   wB(x) = DB(x)^(-r) * P(B|c(x))

   where DF/DB are the distances to the confident foreground/background
   (Euclidean distance transform as a fast approximation of the geodesic
   distance) and r = 2.

4. Matting: J(x) = alpha(x) * c(x) + (1 - alpha(x)) * B_new(x). The band is
   narrow, so the color-bleed effect the lecture warns about is negligible.

The alpha video stores alpha in [0, 1] scaled to [0, 255] uint8 (FAQ #5),
replicated to 3 channels.
"""
import time

import cv2
import numpy as np

from utils import open_video, create_writer, frames_of, bbox_of_mask

RHO = 10            # half-width of the undecided band around the contour
SAMPLE_RADIUS = 25  # how far from the band color samples are collected
KDE_SIGMA = 6.0     # Parzen bell width (histogram bins = uint8 units)
DIST_R = 2.0        # exponent of the distance weighting
EPS = 1e-8

_ERODE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RHO + 1, 2 * RHO + 1))
_SAMPLE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * SAMPLE_RADIUS + 1, 2 * SAMPLE_RADIUS + 1))


def _kde_lut(samples_1d):
    """256-entry KDE lookup table from uint8 samples (histogram + Gaussian)."""
    hist = np.bincount(samples_1d, minlength=256).astype(np.float32)
    lut = cv2.GaussianBlur(hist.reshape(-1, 1), (0, 0), KDE_SIGMA).ravel()
    lut /= (lut.sum() + EPS)
    return lut + EPS


def _posterior_fg(lab_roi, fg_samples, bg_samples):
    """P(F|c) per pixel from per-channel KDE color models (uniform priors)."""
    log_f = np.zeros(lab_roi.shape[:2], np.float32)
    log_b = np.zeros(lab_roi.shape[:2], np.float32)
    for c in range(3):
        lut_f = _kde_lut(fg_samples[:, c])
        lut_b = _kde_lut(bg_samples[:, c])
        ch = lab_roi[..., c]
        log_f += np.log(lut_f)[ch]
        log_b += np.log(lut_b)[ch]
    m = np.maximum(log_f, log_b)
    pf = np.exp(log_f - m)
    pb = np.exp(log_b - m)
    return pf / (pf + pb)


def _alpha_for_frame(frame, mask):
    """Full-frame float32 alpha map in [0, 1] for one stabilized frame."""
    h, w = mask.shape
    alpha = np.zeros((h, w), np.float32)
    box = bbox_of_mask(mask)
    if box is None:
        return alpha
    r0, c0, bh, bw = box
    m = SAMPLE_RADIUS + RHO + 5
    r0, r1 = max(0, r0 - m), min(h, r0 + bh + m)
    c0, c1 = max(0, c0 - m), min(w, c0 + bw + m)

    mask_roi = mask[r0:r1, c0:c1]
    fg_core = cv2.erode(mask_roi, _ERODE_K)
    dilated = cv2.dilate(mask_roi, _ERODE_K)
    band = (dilated - fg_core).astype(bool)

    alpha_roi = fg_core.astype(np.float32)
    if band.any():
        lab_roi = cv2.cvtColor(frame[r0:r1, c0:c1], cv2.COLOR_BGR2Lab)
        near_band = cv2.dilate(dilated - fg_core, _SAMPLE_K).astype(bool)
        fg_sample_px = (fg_core == 1) & near_band
        bg_sample_px = (dilated == 0) & near_band
        if fg_sample_px.any() and bg_sample_px.any():
            p_fg = _posterior_fg(lab_roi, lab_roi[fg_sample_px], lab_roi[bg_sample_px])
            # distance maps to the confident regions (lecture: D_F, D_B)
            d_f = cv2.distanceTransform(1 - fg_core, cv2.DIST_L2, 3)
            d_b = cv2.distanceTransform(dilated, cv2.DIST_L2, 3)
            w_f = np.power(d_f + 1.0, -DIST_R) * p_fg
            w_b = np.power(d_b + 1.0, -DIST_R) * (1.0 - p_fg)
            alpha_band = w_f / (w_f + w_b + EPS)
            alpha_roi[band] = alpha_band[band]
            alpha_roi = cv2.GaussianBlur(alpha_roi, (5, 5), 0)
            alpha_roi[fg_core == 1] = 1.0
            alpha_roi[dilated == 0] = 0.0

    alpha[r0:r1, c0:c1] = alpha_roi
    return alpha


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
        alpha = _alpha_for_frame(frame, mask)

        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        alpha_writer.write(cv2.merge([alpha_u8, alpha_u8, alpha_u8]))

        a3 = alpha[..., None]
        matted = a3 * frame.astype(np.float32) + (1.0 - a3) * background_f
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
