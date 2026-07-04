"""Stage 1 - Video stabilization.

We remove camera shake with the Lucas-Kanade optical-flow method taught in the
course (ported from ex2 ``lucas_kanade.py``), specialised to a *global
translation* motion model:

    1. For each consecutive pair of frames estimate a single (u, v) shift that
       best explains the motion, using a coarse-to-fine LK solve.  Instead of a
       per-pixel flow we sum the LK normal equations over the whole frame, which
       yields one translation for the entire image - exactly the dominant term
       of camera jitter.
    2. Integrate those pairwise shifts into a camera trajectory and compensate it.
       This clip is a hand-held but essentially *static* camera (the background is
       fixed, only the person walks), so we lock every frame to the average
       position: correction = mean(path) - path.  That cancels the jitter
       completely and yields a background that is static across the whole clip -
       exactly what the next stage (background subtraction) needs, while keeping
       the black borders minimal.  (Set ``STAB_SMOOTH_RADIUS > 0`` for a
       moving-average path instead, which would preserve a slow intentional pan.)
    3. Warp every (colour) frame by the correction shift with ``cv2.warpAffine``.
       All three channels get the *same* shift, and we use a constant black
       border - the brief explicitly allows black edges and warns against the
       interpolation blur that a crop-and-resize would add.

Only the motion *estimation* is downscaled (for speed); the *warp* is applied at
full resolution so no sharpness is lost before the matting stage.
"""
import cv2
import numpy as np

import config as cfg


def _lk_translation(prev_gray, cur_gray, num_levels, max_iter):
    """Estimate the global (u, v) translation moving `prev` onto `cur`.

    Coarse-to-fine Lucas-Kanade with a single (u, v) per level: at each level we
    warp `cur` by the running estimate, build the summed structure tensor
    [[Sxx, Sxy], [Sxy, Syy]] and solve the 2x2 system for the increment
    (du, dv).  Convention: a point at x in `prev` appears at x+u in `cur`.
    """
    prev_gray = prev_gray.astype(np.float32)
    cur_gray = cur_gray.astype(np.float32)

    pyr_prev = [prev_gray]
    pyr_cur = [cur_gray]
    for _ in range(num_levels):
        pyr_prev.append(cv2.pyrDown(pyr_prev[-1]))
        pyr_cur.append(cv2.pyrDown(pyr_cur[-1]))

    u = v = 0.0
    for level in range(num_levels, -1, -1):
        I1 = pyr_prev[level]
        I2 = pyr_cur[level]
        h, w = I1.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)
        Sxx = float(np.sum(Ix * Ix))
        Syy = float(np.sum(Iy * Iy))
        Sxy = float(np.sum(Ix * Iy))
        det = Sxx * Syy - Sxy * Sxy
        if abs(det) < 1e-6:
            if level > 0:
                u *= 2.0
                v *= 2.0
            continue
        for _ in range(max_iter):
            map_x = (gx + u).astype(np.float32)
            map_y = (gy + v).astype(np.float32)
            warped = cv2.remap(I2, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            It = warped - I1
            Sxt = float(np.sum(Ix * It))
            Syt = float(np.sum(Iy * It))
            du = (-Syy * Sxt + Sxy * Syt) / det
            dv = (Sxy * Sxt - Sxx * Syt) / det
            u += du
            v += dv
            if du * du + dv * dv < 1e-4:
                break
        if level > 0:                    # pass the estimate to the finer level
            u *= 2.0
            v *= 2.0
    return u, v


def _smooth_path(path, radius):
    """Moving-average smoothing of a 1-D trajectory with edge padding."""
    if radius < 1:
        return path
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    padded = np.pad(path, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stabilize(frames, params):
    """Stabilize a list of BGR frames.

    Returns a new list of stabilized BGR uint8 frames (same count and size).
    """
    n = len(frames)
    h, w = frames[0].shape[:2]

    scale = cfg.STAB_PROC_WIDTH / float(w)
    small_size = (cfg.STAB_PROC_WIDTH, max(1, int(round(h * scale))))

    grays_small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), small_size)
                   for f in frames]

    # 1. pairwise shifts -> raw camera trajectory (in full-res pixels)
    traj_x = np.zeros(n)
    traj_y = np.zeros(n)
    for i in range(1, n):
        du, dv = _lk_translation(grays_small[i - 1], grays_small[i],
                                 cfg.STAB_NUM_LEVELS, cfg.STAB_MAX_ITER)
        traj_x[i] = traj_x[i - 1] + du / scale
        traj_y[i] = traj_y[i - 1] + dv / scale

    # 2. compensate the camera path; correction = reference - raw
    if cfg.STAB_SMOOTH_RADIUS > 0:            # follow a smoothed path (keeps pan)
        ref_x = _smooth_path(traj_x, cfg.STAB_SMOOTH_RADIUS)
        ref_y = _smooth_path(traj_y, cfg.STAB_SMOOTH_RADIUS)
    else:                                     # lock to the average position
        ref_x = np.full_like(traj_x, traj_x.mean())
        ref_y = np.full_like(traj_y, traj_y.mean())
    corr_x = ref_x - traj_x
    corr_y = ref_y - traj_y

    # 3. warp each colour frame by its correction shift
    stabilized = []
    for i, f in enumerate(frames):
        M = np.array([[1.0, 0.0, corr_x[i]],
                      [0.0, 1.0, corr_y[i]]], dtype=np.float32)
        warped = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        stabilized.append(warped)
    return stabilized
