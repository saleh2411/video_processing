"""Stage 4 - Tracking.

We track the single walking person and draw a rectangle on every frame.

The tracker is a colour **particle filter** (ported from the course ex3
``particle_filter.py``, with the matplotlib display removed) whose *observation*
is fused with the segmentation mask produced in stage 2:

    * State per particle: [x_center, y_center, half_w, half_h, v_x, v_y].
    * The particles are propagated with a constant-velocity motion model plus
      Gaussian noise and weighted by the Bhattacharyya similarity between their
      patch colour histogram and a reference histogram of the person - this is
      the ex3 particle filter.
    * Because stage 2 already gives a clean per-frame silhouette, we use its
      bounding box as the measurement: when a mask is present it anchors the
      particle cloud (the box is the tight mask box, and the particles are
      re-seeded around it so the filter cannot drift); when a mask is missing the
      colour particle filter coasts on its own estimate.  A light exponential
      smoothing removes single-frame jitter.

This "mask bounding-box" measurement is one of the tracking approaches the
project guide lists, and fusing it with the particle filter gives a box that is
both tight (good against the secret ground-truth boxes) and robust.

The box is recorded as ``[ROW, COL, HEIGHT, WIDTH]`` in tracking.json and drawn
onto the matted frames to produce OUTPUT.
"""
import numpy as np
import cv2

import config as cfg

QUANT = 16                                   # colour bins per channel


def _mask_box(mask):
    """Tight bounding box of the mask as [ROW, COL, HEIGHT, WIDTH], or None."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return [int(ys.min()), int(xs.min()),
            int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)]


def _histogram(image, state):
    """Normalized QUANT^3 colour histogram of the patch described by `state`."""
    xc, yc, hw, hh = int(state[0]), int(state[1]), int(state[2]), int(state[3])
    h_img, w_img = image.shape[:2]
    x0, x1 = max(0, xc - hw), min(w_img, xc + hw)
    y0, y1 = max(0, yc - hh), min(h_img, yc + hh)
    patch = image[y0:y1, x0:x1, :]
    if patch.size == 0:
        return np.zeros(QUANT ** 3, dtype=np.float64)
    q = (patch // (256 // QUANT)).astype(np.int32)
    idx = (q[:, :, 0] * QUANT * QUANT + q[:, :, 1] * QUANT + q[:, :, 2]).ravel()
    hist = np.bincount(idx, minlength=QUANT ** 3).astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else hist


def _predict(state):
    """Constant-velocity drift + Gaussian noise (motion model)."""
    s = state.astype(float).copy()
    n = s.shape[1]
    s[0] += s[4] + np.random.normal(0, cfg.TRK_POS_NOISE, n)
    s[1] += s[5] + np.random.normal(0, cfg.TRK_POS_NOISE, n)
    s[4] += np.random.normal(0, cfg.TRK_VEL_NOISE, n)
    s[5] += np.random.normal(0, cfg.TRK_VEL_NOISE, n)
    return s


def _resample(state, cdf):
    """Systematic resampling of particles according to the CDF."""
    n = state.shape[1]
    out = np.zeros_like(state)
    start = np.random.uniform(0, 1.0 / n)
    points = start + np.arange(n) / n
    j = 0
    for i in range(n):
        while j < n - 1 and cdf[j] < points[i]:
            j += 1
        out[:, i] = state[:, j]
    return out


def _weights(image, state, ref_hist):
    """Bhattacharyya-based weights and their CDF for all particles."""
    n = state.shape[1]
    w = np.zeros(n)
    for i in range(n):
        p = _histogram(image, state[:, i])
        bc = np.sum(np.sqrt(p * ref_hist))          # Bhattacharyya coefficient
        w[i] = np.exp(20 * bc)
    total = w.sum()
    w = w / total if total > 0 else np.ones(n) / n
    return w, np.cumsum(w)


def _mean_box(state, weights):
    """Weighted-mean state -> [ROW, COL, HEIGHT, WIDTH] integer box."""
    xc = np.sum(state[0] * weights)
    yc = np.sum(state[1] * weights)
    hw = np.sum(state[2] * weights)
    hh = np.sum(state[3] * weights)
    return [int(round(yc - hh)), int(round(xc - hw)),
            int(round(2 * hh)), int(round(2 * hw))]


def _reseed(state, box):
    """Re-seed the particle cloud around a measured box (the mask correction)."""
    row, col, bh, bw = box
    n = state.shape[1]
    xc, yc = col + bw / 2.0, row + bh / 2.0
    state[0] = xc + np.random.normal(0, cfg.TRK_POS_NOISE, n)
    state[1] = yc + np.random.normal(0, cfg.TRK_POS_NOISE, n)
    state[2] = bw / 2.0
    state[3] = bh / 2.0
    return state


def track(matted_frames, masks, params, init_box=None):
    """Track the person through `matted_frames`, fusing the stage-2 masks.

    Returns (output_frames with the rectangle drawn, boxes dict).
    """
    n = len(matted_frames)
    N = cfg.TRK_NUM_PARTICLES
    meas = [_mask_box(m) for m in masks]

    first = init_box if init_box is not None else next((b for b in meas if b), None)
    if first is None:                                # nothing ever detected
        h, w = matted_frames[0].shape[:2]
        first = [h // 4, w // 4, h // 2, w // 2]

    row, col, bh, bw = first
    s0 = np.array([col + bw / 2.0, row + bh / 2.0, bw / 2.0, bh / 2.0, 0.0, 0.0])
    state = np.tile(s0.reshape(6, 1), (1, N))
    ref_hist = _histogram(matted_frames[0], s0)
    weights, cdf = _weights(matted_frames[0], state, ref_hist)

    output_frames = []
    boxes = {}
    smoothed = None
    for i, frame in enumerate(matted_frames):
        if i > 0:                                    # particle filter step
            state = _predict(_resample(state, cdf))
            weights, cdf = _weights(frame, state, ref_hist)
            best = _histogram(frame, state[:, int(np.argmax(weights))])
            ref_hist = 0.95 * ref_hist + 0.05 * best

        if meas[i] is not None:                      # anchor to the mask box
            box = meas[i]
            state = _reseed(state, box)
        else:                                        # coast on the colour filter
            box = _mean_box(state, weights)

        # light exponential smoothing to remove single-frame jitter
        if smoothed is None:
            smoothed = list(map(float, box))
        else:
            a = cfg.TRK_SMOOTH
            smoothed = [a * b + (1 - a) * s for b, s in zip(box, smoothed)]
        final = [int(round(v)) for v in smoothed]
        boxes[i + cfg.TRK_KEY_OFFSET] = final

        r, c, hh, ww = final
        out = frame.copy()
        cv2.rectangle(out, (c, r), (c + ww, r + hh),
                      cfg.TRK_BOX_COLOR, cfg.TRK_BOX_THICKNESS)
        output_frames.append(out)

    return output_frames, boxes
