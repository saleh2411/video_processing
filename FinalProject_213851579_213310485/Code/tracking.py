"""Track the person in the matted video using a particle filter (Condensation).

Each particle is a guess for the box: [x_center, y_center, half_width,
half_height, x_velocity, y_velocity]. Every frame we resample particles by
weight, move them, score them against a reference color histogram, and report
the weighted-average box.

The first non-empty binary mask gives the starting box, so no manual init.
Outputs: OUTPUT video with a green tracking box, and tracking.json with
per-frame [ROW, COL, HEIGHT, WIDTH].
"""
import json

from tqdm import tqdm
import utils
import cv2
import numpy as np

N = 100                 # number of particles

POS_NOISE_STD = 10      # position noise added each step, in pixels
VEL_NOISE_STD = 3       # velocity noise added each step, in pixels

HIST_BINS = 16          # 16 bins per BGR channel (4-bit color)
BHATTA_GAIN = 20        # weight = exp(BHATTA_GAIN * similarity)


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Move particles by their velocity and add random noise."""
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()
    n = state_drifted.shape[1]
    # move by velocity
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]
    # add random noise to position and velocity
    state_drifted[0, :] += np.round(np.random.normal(0, POS_NOISE_STD, n))
    state_drifted[1, :] += np.round(np.random.normal(0, POS_NOISE_STD, n))
    state_drifted[4, :] += np.round(np.random.normal(0, VEL_NOISE_STD, n))
    state_drifted[5, :] += np.round(np.random.normal(0, VEL_NOISE_STD, n))

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Normalized color histogram of the box described by state."""
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((HIST_BINS, HIST_BINS, HIST_BINS))
    xc, yc, half_w, half_h = state[0], state[1], state[2], state[3]
    h_img, w_img = image.shape[0], image.shape[1]
    # clip the box to the image
    x_min, x_max = max(0, xc - half_w), min(w_img, xc + half_w)
    y_min, y_max = max(0, yc - half_h), min(h_img, yc + half_h)
    patch = image[y_min:y_max, x_min:x_max, :]
    hist = np.reshape(hist, HIST_BINS ** 3)
    if patch.size == 0:                                  # box fully off-screen
        return hist
    # quantize colors and count them
    q_patch = (patch // HIST_BINS).astype(int)
    idx = (q_patch[:, :, 0] * HIST_BINS ** 2 + q_patch[:, :, 1] * HIST_BINS +
           q_patch[:, :, 2]).ravel()
    hist = np.bincount(idx, minlength=HIST_BINS ** 3).astype(float)
    hist = hist / sum(hist)

    return hist


def compute_reference_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reference histogram from the person pixels only (mask == 1)."""
    person_px = image[mask.astype(bool)]
    q_px = (person_px // HIST_BINS).astype(int)
    idx = q_px[:, 0] * HIST_BINS ** 2 + q_px[:, 1] * HIST_BINS + q_px[:, 2]
    hist = np.bincount(idx, minlength=HIST_BINS ** 3).astype(float)
    hist = hist / sum(hist)
    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Resample particles: heavier particles get picked more often."""
    S_next = np.zeros(previous_state.shape)
    n = previous_state.shape[1]
    for i in range(n):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:, i] = previous_state[:, j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Weight from how similar histograms p and q are."""
    bc = np.sum(np.sqrt(p * q))
    distance = np.exp(BHATTA_GAIN * bc)
    return distance


def compute_weights_and_cdf(image, S, q):
    """Score every particle against q, then normalize and build the CDF."""
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W[i] = bhattacharyya_distance(p, q)
    W = W / np.sum(W)            # make the weights sum to 1
    C = np.cumsum(W)             # running sum, used for resampling
    return W, C


def weighted_mean_box(S, W, frame_h, frame_w):
    """Weighted-average box as [row, col, height, width], clipped to frame."""
    half_w_avg = np.sum(S[2, :] * W)
    half_h_avg = np.sum(S[3, :] * W)
    x_avg = np.sum(S[0, :] * W) - half_w_avg
    y_avg = np.sum(S[1, :] * W) - half_h_avg
    row = int(round(max(0, min(y_avg, frame_h - 1))))
    col = int(round(max(0, min(x_avg, frame_w - 1))))
    box_h = int(round(min(2 * half_h_avg, frame_h - row)))
    box_w = int(round(min(2 * half_w_avg, frame_w - col)))
    return [row, col, box_h, box_w]


def find_initial_state(binary_video_path):
    """First non-empty mask gives the starting box (frame index, state, mask)."""
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
    """Track the person and write the OUTPUT video and tracking.json."""
    # read video info
    cap = utils.open_video(matted_video_path)
    params = utils.get_video_parameters(cap)
    w, h, fps = params["width"], params["height"], params["fps"]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # create output video
    out = utils.create_video(output_video_path, fps, w, h)

    # starting box from the first non-empty mask
    init_idx, s_initial, init_mask = find_initial_state(binary_video_path)

    S = None
    q = None
    C = None
    results = {}

    for i, frame in enumerate(tqdm(utils.frames_of(matted_video_path),
                                   total=frame_count, desc="Tracking")):
        if i < init_idx:
            # person not seen yet - report the future starting box
            box = weighted_mean_box(np.array(s_initial, float)[:, None],
                                    np.array([1.0]), h, w)
        elif i == init_idx:
            # first tracked frame: build reference from person pixels,
            # spread N particles around the starting box
            q = compute_reference_histogram(frame, init_mask)
            S = predict_particles(np.tile(np.array(s_initial), (N, 1)).T)
            W, C = compute_weights_and_cdf(frame, S, q)
            box = weighted_mean_box(S, W, h, w)
        else:
            # main loop: resample -> move -> score
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
