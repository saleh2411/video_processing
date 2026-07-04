"""Stage 2 - Background subtraction.

The video is stabilized, so the scene is nearly static and the only large moving
object is the walking person.  We build a per-pixel statistical background model
and flag pixels that deviate from it far more than that pixel's own natural
variation.

    1. Background model.  From a set of evenly sampled frames we estimate, for
       every pixel, a robust centre (temporal median) and a robust spread (median
       absolute deviation, MAD) in CIE-Lab.  Lab separates luminance (L) from
       chroma (a,b), so shadows - which move L but not a,b - are largely rejected.
       Black stabilization-border pixels are excluded from the statistics.

    2. Adaptive detection.  A pixel is foreground when its Lab deviation from the
       centre, measured in units of that pixel's spread (a Mahalanobis-style
       z-score), exceeds a threshold.  This is the key robustness trick: a
       high-contrast wall region that flickers a little under residual sub-pixel
       misalignment has a large spread, so it needs a large deviation to fire and
       stops leaking into the mask; a flat wall has a tiny spread, so the person
       is still detected against it.

    3. Clean-up.  Opening removes speckle, closing fills holes, we keep the single
       largest connected component (one person, FAQ #6) and fill its interior.

Outputs (frozen contract):
    extracted_frames : person's real colour pixels, 0 elsewhere.
    binary_masks     : uint8 HxW in {0,1}, 1 = person.
"""
import cv2
import numpy as np

import config as cfg


def _background_model(frames):
    """Per-pixel robust centre and spread in Lab from sampled frames.

    Returns (centre_lab HxWx3 float32, spread_lab HxWx3 float32).
    """
    n = len(frames)
    k = min(cfg.BGS_MEDIAN_SAMPLES, n)
    idx = np.linspace(0, n - 1, k).astype(int)

    labs = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
                     for i in idx], axis=0)
    bgr = np.stack([frames[i] for i in idx], axis=0)
    border = np.all(bgr <= cfg.BGS_BORDER_BLACK, axis=3)   # K,H,W
    labs[border] = np.nan

    centre = np.nanmedian(labs, axis=0)                    # H,W,3 robust centre
    # MAD (not std) for the spread: it ignores the person-passing outliers, so a
    # pixel the person walks through keeps a small spread and is still detectable.
    mad = np.nanmedian(np.abs(labs - centre), axis=0)      # H,W,3
    spread = 1.4826 * mad                                  # MAD -> ~sigma
    centre = np.nan_to_num(centre, nan=0.0)
    spread = np.nan_to_num(spread, nan=0.0)
    # floor keeps flat, noise-free regions sensitive (avoids divide-by-zero)
    spread = np.maximum(spread, np.array([6.0, 4.0, 4.0], np.float32))
    return centre, spread


def _largest_component(mask):
    """Keep only the largest connected foreground blob."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def _fill_holes(mask):
    """Fill holes fully enclosed by the foreground via flood fill from a corner."""
    h, w = mask.shape
    ff = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    holes = 1 - ff
    return ((mask | holes) > 0).astype(np.uint8)


def subtract(stab_frames, params):
    h, w = stab_frames[0].shape[:2]
    area = h * w
    min_area = cfg.BGS_MIN_AREA_FRAC * area

    centre, spread = _background_model(stab_frames)

    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.BGS_OPEN_KERNEL, cfg.BGS_OPEN_KERNEL))
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.BGS_CLOSE_KERNEL, cfg.BGS_CLOSE_KERNEL))

    extracted_frames = []
    binary_masks = []
    for f in stab_frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        z = (lab - centre) / spread                # per-channel z-score
        z2 = np.sum(z * z, axis=2)                  # Mahalanobis (diagonal) distance^2

        # ignore the stabilization border, eroded inward so its seam cannot fire
        valid = (np.any(f > cfg.BGS_BORDER_BLACK, axis=2)).astype(np.uint8)
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
        z2[valid == 0] = 0.0

        mask = (z2 > cfg.BGS_Z_THRESH ** 2).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = _largest_component(mask)
        if int(mask.sum()) < min_area:
            mask = np.zeros((h, w), np.uint8)
        else:
            mask = _fill_holes(mask)

        binary_masks.append(mask)
        extracted_frames.append(f * mask[:, :, None])

    return extracted_frames, binary_masks
