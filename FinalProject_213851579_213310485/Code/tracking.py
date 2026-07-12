"""Tracking (lecture: particle filter / Condensation; algorithm of ex3).

Final pipeline step: the person is tracked in the matted video with the
particle filter implemented in ex3 (HW3), i.e. the Condensation algorithm of
the tracking lecture. No manual initialization is needed: the first non-empty
mask of the binary video (the background-subtraction output) gives the
initial bounding box, so the tracker bootstraps itself from earlier pipeline
stages.

State of a particle (as in ex3):

    s = [x_center, y_center, half_width, half_height, x_velocity, y_velocity]

One time step of the Condensation loop (lecture / ex3 PDF steps):

1. SAMPLE:  draw N particles from the previous set according to the weights,
   by inverting the CDF C (a uniform r in [0,1] picks the first j with
   C[j] >= r) - particles with high likelihood are duplicated, unlikely ones
   die out (importance resampling).
2. PREDICT (drift + diffuse): apply the constant-velocity motion model
   x += vx, y += vy, then add white Gaussian noise to position and velocity
   (the system dynamics s_t = A s_{t-1} + w_t of the lecture).
3. MEASURE: each particle's rectangle is described by a normalized joint
   16x16x16 color histogram (color quantized to 4 bits per channel, as in
   ex3); its similarity to the reference histogram q of the initial box is
   the Bhattacharyya coefficient BC(p, q) = sum sqrt(p*q), and the particle
   weight is w = exp(20 * BC) (the likelihood used in ex3).
4. NORMALIZE the weights so sum(W) = 1 and build the CDF C = cumsum(W) for
   the next sampling step.
5. ESTIMATE: the output box is the weighted MEAN state E[s] = sum W_i * s_i
   (the "average particle box" of ex3).

Outputs: OUTPUT video = matted video with the tracked box drawn in green, and
tracking.json with per-frame entries "1".."N" -> [ROW, COL, HEIGHT, WIDTH]
(top-left row/col) per the project FAQ.
"""
import json

from tqdm import tqdm
import utils
import cv2
import numpy as np

# SET NUMBER OF PARTICLES (as in ex3)
N = 100

# diffusion noise of the predict step, in pixels. ex3 used (4, 1) on small
# images; here the video is full-HD and the person walks ~10 px/frame (and
# turns around mid-video), so the diffusion is scaled up accordingly - the
# ex3 PDF explicitly leaves the noise magnitudes as a free choice.
POS_NOISE_STD = 10
VEL_NOISE_STD = 3

HIST_BINS = 16          # 16 bins per BGR channel = 4-bit quantization (ex3)
BHATTA_GAIN = 20        # weight = exp(BHATTA_GAIN * BC), as in ex3


# ex3 / lecture rule (PREDICT, drift + diffuse): system dynamics
# s_t = A s_{t-1} + w_t with A the constant-velocity motion model
# (x += vx, y += vy) and w_t white Gaussian noise on position and velocity.
def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise (ex3)."""
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()
    n = state_drifted.shape[1]
    # drift: constant-velocity motion model
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]
    # diffuse: white Gaussian noise on position and velocity
    state_drifted[0, :] += np.round(np.random.normal(0, POS_NOISE_STD, n))
    state_drifted[1, :] += np.round(np.random.normal(0, POS_NOISE_STD, n))
    state_drifted[4, :] += np.round(np.random.normal(0, VEL_NOISE_STD, n))
    state_drifted[5, :] += np.round(np.random.normal(0, VEL_NOISE_STD, n))

    state_drifted = state_drifted.astype(int)
    return state_drifted


# ex3 / lecture rule (MEASURE, observation model): a candidate box is
# described by its normalized joint color histogram with 4-bit quantization
# per channel - 16x16x16 bins - so the appearance model is illumination- and
# deformation-tolerant, exactly as in ex3.
def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters (ex3)."""
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((HIST_BINS, HIST_BINS, HIST_BINS))
    xc, yc, half_w, half_h = state[0], state[1], state[2], state[3]
    h_img, w_img = image.shape[0], image.shape[1]
    # clip the rectangle to the image so we never index outside it
    x_min, x_max = max(0, xc - half_w), min(w_img, xc + half_w)
    y_min, y_max = max(0, yc - half_h), min(h_img, yc + half_h)
    patch = image[y_min:y_max, x_min:x_max, :]
    hist = np.reshape(hist, HIST_BINS ** 3)
    if patch.size == 0:                                  # particle fully off-screen
        return hist
    # quantize each channel to 4 bits and count joint (B, G, R) bins
    q_patch = (patch // HIST_BINS).astype(int)
    idx = (q_patch[:, :, 0] * HIST_BINS ** 2 + q_patch[:, :, 1] * HIST_BINS +
           q_patch[:, :, 2]).ravel()
    hist = np.bincount(idx, minlength=HIST_BINS ** 3).astype(float)
    # normalize so the histogram is a probability distribution
    hist = hist / sum(hist)

    return hist


# Lecture rule (target model of color-histogram tracking): the reference
# distribution q is the color histogram of the OBJECT pixels, not of a full
# rectangle. The binary mask of the background-subtraction stage marks the
# person's pixels exactly, so q is estimated from them alone - a box that
# merely covers background cannot score a high Bhattacharyya coefficient.
def compute_reference_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized 16^3 color histogram of the person pixels (mask == 1)."""
    person_px = image[mask.astype(bool)]
    q_px = (person_px // HIST_BINS).astype(int)
    idx = q_px[:, 0] * HIST_BINS ** 2 + q_px[:, 1] * HIST_BINS + q_px[:, 2]
    hist = np.bincount(idx, minlength=HIST_BINS ** 3).astype(float)
    hist = hist / sum(hist)
    return hist


# ex3 / lecture rule (SAMPLE, importance resampling): draw each new particle
# by inverting the CDF - a uniform r picks the first index j with C[j] >= r,
# so particles are reproduced in proportion to their weights.
def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf (ex3)."""
    S_next = np.zeros(previous_state.shape)
    n = previous_state.shape[1]
    for i in range(n):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:, i] = previous_state[:, j]
    return S_next


# ex3 / lecture rule (likelihood): the Bhattacharyya coefficient
# BC(p, q) = sum sqrt(p * q) measures the overlap of two color
# distributions; the exponential turns it into the particle likelihood.
def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya-based likelihood between histograms p and q (ex3)."""
    bc = np.sum(np.sqrt(p * q))
    distance = np.exp(BHATTA_GAIN * bc)
    return distance


# ex3 / lecture rule (steps 5-6 of the PDF): normalize the weights to a
# probability distribution (sum W = 1) and build the CDF C = cumsum(W) used
# by the next resampling step.
def compute_weights_and_cdf(image, S, q):
    """Score each particle's patch against the reference histogram q (ex3)."""
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W[i] = bhattacharyya_distance(p, q)
    W = W / np.sum(W)            # normalize: sum(W) == 1   (PDF step 5)
    C = np.cumsum(W)             # CDF: C[j] = sum of W[0..j]  (PDF step 6)
    return W, C


# ex3 rule (ESTIMATE): the tracked box is the weighted mean of the particle
# set, E[s] = sum W_i * s_i - the green "average particle box" of ex3.
def weighted_mean_box(S, W, frame_h, frame_w):
    """[row, col, height, width] of the weighted-mean particle, clipped."""
    half_w_avg = np.sum(S[2, :] * W)
    half_h_avg = np.sum(S[3, :] * W)
    x_avg = np.sum(S[0, :] * W) - half_w_avg
    y_avg = np.sum(S[1, :] * W) - half_h_avg
    row = int(round(max(0, min(y_avg, frame_h - 1))))
    col = int(round(max(0, min(x_avg, frame_w - 1))))
    box_h = int(round(min(2 * half_h_avg, frame_h - row)))
    box_w = int(round(min(2 * half_w_avg, frame_w - col)))
    return [row, col, box_h, box_w]


# Automatic initialization (replaces ex3's hard-coded s_initial): the first
# non-empty mask of the binary video gives the initial box, so the tracker is
# initialized by the background-subtraction stage instead of by hand.
def find_initial_state(binary_video_path):
    """(frame_index, s_initial, mask) from the first non-empty binary mask."""
    for i, binary_frame in enumerate(utils.frames_of(binary_video_path)):
        mask = (binary_frame[..., 0] > 127).astype(np.uint8)
        box = utils.bbox_of_mask(mask)
        if box is not None:
            r0, c0, bh, bw = box
            s_initial = [c0 + bw // 2,    # x center
                         r0 + bh // 2,    # y center
                         bw // 2,         # half width
                         bh // 2,         # half height
                         0,               # velocity x
                         0]               # velocity y
            return i, s_initial, mask
    raise ValueError(f"No person found in binary video {binary_video_path}")


def tracking(matted_video_path, binary_video_path, output_video_path,
             tracking_json_path):
    """Write the OUTPUT video (green box around the tracked person) and
    tracking.json, using the ex3 particle filter on the matted video."""
    # open video and get params
    cap = utils.open_video(matted_video_path)
    params = utils.get_video_parameters(cap)
    w, h, fps = params["width"], params["height"], params["fps"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # create output video
    out = utils.create_video(output_video_path, fps, w, h)

    # initial state from the background-subtraction mask (no manual init)
    init_idx, s_initial, init_mask = find_initial_state(binary_video_path)

    S = None
    q = None
    C = None
    results = {}

    for i, frame in enumerate(tqdm(utils.frames_of(matted_video_path),
                                   total=frame_count, desc="Tracking")):
        if i < init_idx:
            # person not detected yet - report the (future) initial box
            box = weighted_mean_box(np.array(s_initial, float)[:, None],
                                    np.array([1.0]), h, w)
        elif i == init_idx:
            # ex3 initialization, with the target model q taken from the
            # person pixels (binary mask) instead of the whole initial box;
            # then spread N copies of s_initial with one predict step
            q = compute_reference_histogram(frame, init_mask)
            S = predict_particles(np.tile(np.array(s_initial), (N, 1)).T)
            W, C = compute_weights_and_cdf(frame, S, q)
            box = weighted_mean_box(S, W, h, w)
        else:
            # ex3 main tracking loop: SAMPLE -> PREDICT -> MEASURE
            S = predict_particles(sample_particles(S, C))
            W, C = compute_weights_and_cdf(frame, S, q)
            box = weighted_mean_box(S, W, h, w)

        # tracking.json entry [ROW, COL, HEIGHT, WIDTH], keys start at "1"
        results[str(i + 1)] = box
        row, col, box_h, box_w = box
        cv2.rectangle(frame, (col, row), (col + box_w, row + box_h),
                      (0, 255, 0), 3)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    with open(tracking_json_path, 'w') as f:
        json.dump(results, f, indent=4)
