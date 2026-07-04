"""Background subtraction (lecture: per-pixel background modeling).

Method
------
After stabilization the background is static, so each pixel sees its true
background color most of the time and the walking person only briefly.
Following the per-pixel background-model idea from the lecture (each pixel is
modeled by the Gaussian it "usually" shows; a new value is foreground when it
falls too many standard deviations away), we build a single dominant Gaussian
per pixel with ROBUST estimators, which are immune to the person passing by:

* mean  <- temporal MEDIAN of the pixel over the (sampled) stabilized video
* sigma <- 1.4826 * MAD (median absolute deviation) - the robust equivalent
           of the standard deviation.

Working color space is CIE-Lab: L carries illumination (where cast shadows
live), a/b carry chroma. The decision score down-weights L so that shadows
(which darken L but barely move chroma) are not classified as foreground:

    score = sqrt(wL * zL^2 + za^2 + zb^2),   z = |x - median| / sigma

A pixel is foreground when score > FG_THRESH (the lecture's "more than ~2.5
sigma away from the background Gaussian" test, applied jointly on the three
channels). Pixels that pass the threshold but look like a CAST SHADOW of the
person are then removed: a shadow only attenuates the luminance of the
background (0.35 < L/L_background < 0.95) while keeping the chroma (a, b)
almost unchanged - the same rule used by OpenCV's MOG2 shadow detection.
The raw mask is cleaned with morphology, the largest connected component is
kept (there is exactly one person, FAQ #6) and its holes are filled.

Outputs: binary_...avi (person = 255 white, else black, saved as color) and
extracted_...avi (stabilized frame * mask: person keeps its original values,
0 everywhere else).
"""
import time

import cv2
import numpy as np

from utils import open_video, create_writer, frames_of, to_color, \
    largest_component, fill_holes

MAX_MODEL_SAMPLES = 60   # frames sampled to build the background model
SIGMA_FLOOR = 2.5        # minimal per-pixel std (uint8 units), avoids z blow-up
L_WEIGHT = 0.25          # weight of the L (luma) z^2 -> shadow suppression
FG_THRESH = 4.5          # joint z-score threshold
OPEN_RADIUS = 5          # remove isolated specks
CLOSE_RADIUS = 21        # glue person parts together
MIN_VALID_L = 5.0        # model is invalid where the median is warp-border black
SHADOW_RATIO_LO = 0.35   # cast shadow: luminance attenuation bounds ...
SHADOW_RATIO_HI = 0.95
SHADOW_CHROMA_MAX = 8.0  # ... while chroma (a, b) stays almost unchanged


def _build_background_model(stabilized_path, frame_count):
    """Per-pixel robust Gaussian (median, sigma) in Lab, from sampled frames."""
    step = max(1, frame_count // MAX_MODEL_SAMPLES)
    samples = []
    for i, frame in enumerate(frames_of(stabilized_path)):
        if i % step == 0:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
    stack = np.stack(samples, axis=0)  # N x H x W x 3, uint8

    median = np.empty(stack.shape[1:], np.float32)
    sigma = np.empty(stack.shape[1:], np.float32)
    for c in range(3):  # channel by channel to limit peak memory
        ch = stack[..., c].astype(np.float32)
        med = np.median(ch, axis=0)
        mad = np.median(np.abs(ch - med), axis=0)
        median[..., c] = med
        sigma[..., c] = np.maximum(1.4826 * mad, SIGMA_FLOOR)
    return median, sigma


def _foreground_mask(frame_bgr, median, sigma):
    """0/1 mask of the walking person in one stabilized frame."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    z = np.abs(lab - median) / sigma
    score = np.sqrt(L_WEIGHT * z[..., 0] ** 2 + z[..., 1] ** 2 + z[..., 2] ** 2)

    mask = (score > FG_THRESH).astype(np.uint8)

    # cast-shadow suppression: darker luminance, nearly unchanged chroma
    ratio = lab[..., 0] / np.maximum(median[..., 0], 1.0)
    chroma_close = (np.abs(lab[..., 1] - median[..., 1]) < SHADOW_CHROMA_MAX) & \
                   (np.abs(lab[..., 2] - median[..., 2]) < SHADOW_CHROMA_MAX)
    shadow = (ratio > SHADOW_RATIO_LO) & (ratio < SHADOW_RATIO_HI) & chroma_close
    mask[shadow] = 0

    # warp borders: ignore pixels that are black in the frame or in the model
    mask[frame_bgr.sum(axis=2) == 0] = 0
    mask[median[..., 0] < MIN_VALID_L] = 0

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_RADIUS, OPEN_RADIUS))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_RADIUS, CLOSE_RADIUS))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    mask = (cv2.medianBlur(mask * 255, 9) > 127).astype(np.uint8)
    return mask


def background_subtraction(stabilized_path, binary_path, extracted_path):
    """Write binary and extracted videos. Returns the wall-clock time at which
    the binary video finished being filled with all its frames."""
    cap, prm = open_video(stabilized_path)
    w, h, fps, n = prm['width'], prm['height'], prm['fps'], prm['frame_count']
    cap.release()

    print('  background subtraction: building per-pixel background model...')
    median, sigma = _build_background_model(stabilized_path, n)

    binary_writer = create_writer(binary_path, fps, w, h)
    extracted_writer = create_writer(extracted_path, fps, w, h)

    n_done = 0
    for frame in frames_of(stabilized_path):
        mask = _foreground_mask(frame, median, sigma)
        binary_writer.write(to_color(mask * 255))
        extracted_writer.write(frame * mask[..., None])
        n_done += 1
        if n_done % 50 == 0:
            print('  background subtraction: {} frames done'.format(n_done))

    binary_writer.release()
    binary_done_time = time.time()
    extracted_writer.release()
    print('  background subtraction: finished ({} frames)'.format(n_done))
    return binary_done_time
