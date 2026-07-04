"""Stage 3 - Image matting and compositing.

Goal: replace the original background with ``background.jpg`` while keeping the
person's edges soft, so the cut-out does not look pasted.  We build a soft alpha
matte alpha in [0,1] and composite:

    out = alpha * foreground + (1 - alpha) * newBackground

The alpha is estimated from the binary mask with a trimap + distance-transform
scheme (fast, no learning, fully explainable):

    1. Trimap.  Eroding the mask gives "definitely foreground" (alpha=1);
       everything outside a dilated mask is "definitely background" (alpha=0).
       The thin ring between them is the "unknown" band where the true edge and
       any soft/motion-blurred hair-like pixels live.
    2. Alpha in the unknown band.  For each unknown pixel let d_fg be the
       distance to the nearest definite-foreground pixel and d_bg the distance to
       the nearest definite-background pixel (both from ``cv2.distanceTransform``).
       Set alpha = d_bg / (d_fg + d_bg): 1 next to the foreground, 0 next to the
       background, a smooth ramp across the band.
    3. A small Gaussian blur feathers the result so the composite edge is smooth
       rather than a hard staircase.

The foreground colour is taken straight from the stabilized frame (the person's
real pixels), and the new background is ``background.jpg`` resized to the frame
size (aspect ratio may change per the brief).
"""
import cv2
import numpy as np

import config as cfg


def _alpha_from_mask(mask, fg_k, bg_k, blur):
    """Trimap + distance-transform soft alpha from a 0/1 mask."""
    mask = (mask > 0).astype(np.uint8)
    fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_k, fg_k))
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_k, bg_k))

    sure_fg = cv2.erode(mask, fg_kernel)
    sure_bg = 1 - cv2.dilate(mask, bg_kernel)
    unknown = 1 - sure_fg - sure_bg            # the band to solve

    # distance from every pixel to the nearest sure-FG / sure-BG pixel
    d_fg = cv2.distanceTransform(1 - sure_fg, cv2.DIST_L2, 3)
    d_bg = cv2.distanceTransform(1 - sure_bg, cv2.DIST_L2, 3)
    band = d_bg / (d_fg + d_bg + 1e-6)

    alpha = sure_fg.astype(np.float32)
    alpha[unknown > 0] = band[unknown > 0]
    if blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
    return np.clip(alpha, 0.0, 1.0)


def matte(stab_frames, binary_masks, bg_image, params):
    """Composite the masked person onto ``bg_image``.

    Returns (matted_frames BGR uint8, alpha_frames uint8 HxW in [0,255]).
    Alpha is stored quantised to uint8 - that is exactly how it must be saved
    (FAQ #5) and it keeps the in-memory footprint small.
    """
    h, w = stab_frames[0].shape[:2]
    new_bg = cv2.resize(bg_image, (w, h)).astype(np.float32)

    matted_frames = []
    alpha_frames = []
    for f, m in zip(stab_frames, binary_masks):
        alpha = _alpha_from_mask(m, cfg.MAT_FG_ERODE, cfg.MAT_BG_DILATE,
                                 cfg.MAT_ALPHA_BLUR)
        a3 = alpha[:, :, None]
        comp = a3 * f.astype(np.float32) + (1.0 - a3) * new_bg
        matted_frames.append(np.clip(comp, 0, 255).astype(np.uint8))
        alpha_frames.append((alpha * 255).astype(np.uint8))
    return matted_frames, alpha_frames
