from tqdm import tqdm
import utils
import cv2
import numpy as np

# Feature detection parameters (Shi-Tomasi corners)
FEATURE_PARAMS = dict(maxCorners=500,
                      qualityLevel=0.01,
                      minDistance=20,
                      blockSize=3)

# Pyramidal Lucas-Kanade parameters (same algorithm as in ex2,
# using OpenCV's implementation for runtime reasons)
LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           30, 0.01))


def estimate_frame_transform(prev_gray, cur_gray):
    identity = np.eye(3, dtype=np.float64)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **FEATURE_PARAMS)
    if prev_pts is None or len(prev_pts) < 6:
        return identity

    cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_pts, None, **LK_PARAMS)
    if cur_pts is None:
        return identity

    valid = status.ravel() == 1
    prev_pts = prev_pts[valid]
    cur_pts = cur_pts[valid]
    if len(prev_pts) < 6:
        return identity

    # maps cur -> prev coordinates
    m, _ = cv2.estimateAffinePartial2D(cur_pts, prev_pts,
                                       method=cv2.RANSAC,
                                       ransacReprojThreshold=3.0)
    if m is None:
        return identity

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :] = m
    return transform


def stabilize_video(input_video_path, output_video_path):

    #open video and get params
    cap = utils.open_video(input_video_path)
    params = utils.get_video_parameters(cap)
    w, h = params["width"], params["height"]

    #create output video
    out = utils.create_video(output_video_path, params["fps"], w, h)

    # Read the first frame, use it as the reference frame for stabilization
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise ValueError("Could not read the first frame from the video.")

    # first frame is the reference - written unchanged
    out.write(first_frame)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    cumulative_transform = np.eye(3, dtype=np.float64)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count - 1), desc="Stabilization"):
        ret, frame = cap.read()
        if not ret:
            break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cur -> prev, composed onto prev -> first gives cur -> first
        transform = estimate_frame_transform(prev_gray, cur_gray)
        cumulative_transform = cumulative_transform @ transform

        stabilized = cv2.warpAffine(frame, cumulative_transform[:2, :], (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        out.write(stabilized)

        prev_gray = cur_gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    