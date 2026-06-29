import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "213851579"
ID2 = "213310485"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()
    n = state_drifted.shape[1]
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]
    state_drifted[0, :] += np.round(np.random.normal(0, 4, n))   
    state_drifted[1, :] += np.round(np.random.normal(0, 4, n))   
    state_drifted[4, :] += np.round(np.random.normal(0, 1, n))   
    state_drifted[5, :] += np.round(np.random.normal(0, 1, n))

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))
    xc, yc, half_w, half_h = state[0], state[1], state[2], state[3]
    h_img, w_img = image.shape[0], image.shape[1]
    # clip the rectangle to the image so we never index outside it
    x_min, x_max = max(0, xc - half_w), min(w_img, xc + half_w)
    y_min, y_max = max(0, yc - half_h), min(h_img, yc + half_h)
    patch = image[y_min:y_max, x_min:x_max, :]
    hist = np.reshape(hist, 16 * 16 * 16)
    if patch.size == 0:                                  # particle fully off-screen
        return np.reshape(hist, 16 * 16 * 16)
    q_patch = (patch // 16).astype(int)
    idx = (q_patch[:, :, 0] * 256 + q_patch[:, :, 1] * 16 + q_patch[:, :, 2]).ravel()
    hist = np.bincount(idx, minlength=16 * 16 * 16).astype(float)
    # normalize
    hist = hist/sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    n = previous_state.shape[1]
    for i in range(n):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)              
        S_next[:, i] = previous_state[:, j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    bc = np.sum(np.sqrt(p * q))
    distance = np.exp(20 * bc)
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    half_w_avg = np.sum(state[2, :] * W)
    half_h_avg = np.sum(state[3, :] * W)
    w_avg = 2 * half_w_avg
    h_avg = 2 * half_h_avg
    x_avg = np.sum(state[0, :] * W) - half_w_avg
    y_avg = np.sum(state[1, :] * W) - half_h_avg


    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    i_max = np.argmax(W)
    w_max = 2 * state[2, i_max]
    h_max = 2 * state[3, i_max]
    x_max = state[0, i_max] - state[2, i_max]
    y_max = state[1, i_max] - state[3, i_max]


    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def compute_weights_and_cdf(image, S, q, N):
    """Score each particle's patch against the reference histogram q."""
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W[i] = bhattacharyya_distance(p, q)
    W = W / np.sum(W)            # normalize: sum(W) == 1   (PDF step 5)
    C = np.cumsum(W)             # CDF: C[j] = sum of W[0..j]  (PDF step 6)
    return W, C


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    W, C = compute_weights_and_cdf(image, S, q, N)
    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W, C = compute_weights_and_cdf(current_image, S, q, N)
        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
