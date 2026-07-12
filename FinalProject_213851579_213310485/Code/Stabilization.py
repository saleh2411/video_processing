"""Video stabilization: anchored KLT tracking + robust homography fit.

Method
------
Every output frame is aligned to the FIRST frame's coordinate system, so
the background becomes static for the background-subtraction stage.

Corners are tracked continuously through the video with pyramidal
Lucas-Kanade (KLT chains) - same algorithm as ex2's lucas_kanade_optical_flow,
OpenCV's implementation for runtime. Each tracked corner carries an
"anchor" - the frame-0 position it corresponds to. Every frame we robustly
fit (RANSAC) a homography mapping current corner positions DIRECTLY onto
their frame-0 anchors:

  - no per-frame transform composition -> estimation errors cannot
    accumulate -> no drift by construction
  - the RANSAC inlier mask prunes points that stopped following the
    global camera motion (corners on the walking person die quickly)
  - when the pool runs low, fresh corners are detected away from the
    surviving ones; their anchors are their positions mapped to frame-0
    coordinates through the current fit
  - a homography (8 DOF) is used because hand shake includes small
    out-of-plane rotations: the residual between frames contains a
    perspective (keystone) component that rotation+translation+scale
    models cannot represent

Black border pixels after warping are expected and allowed (FAQ #1).
Frames are warped exactly once (bilinear), so no quality is lost for
the matting stage.
"""
from tqdm import tqdm
import utils
import cv2
import numpy as np


# Shi-Tomasi corner detection parameters
FEATURE_PARAMS = dict(maxCorners=600,
                      qualityLevel=0.01,
                      minDistance=20,
                      blockSize=11)

# Pyramidal Lucas-Kanade parameters
LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           30, 0.01))

MIN_POINTS = 200        # replenish the tracked pool below this size
RANSAC_THRESH = 3.0     # inlier reprojection threshold [pixels]


def detect_corners(gray, existing=None):
    """Shi-Tomasi corners, avoiding a neighborhood around existing points."""
    mask = None
    if existing is not None and len(existing) > 0:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x, y in existing.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), FEATURE_PARAMS['minDistance'],
                       0, -1)
    return cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)


def stabilize_video(input_video_path, output_video_path):
    """Stabilize the video onto its first frame and write it to file."""
    cap = utils.open_video(input_video_path)
    params = utils.get_video_parameters(cap)
    w, h = params["width"], params["height"]
    out = utils.create_video(output_video_path, params["fps"], w, h)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise ValueError("Could not read the first frame from the video.")

    # first frame is the reference - written unchanged
    out.write(first_frame)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # tracked pool: cur_pts in the current frame, anchors = frame-0 coords
    cur_pts = detect_corners(prev_gray)
    anchors = cur_pts.copy()

    transform = np.eye(3, dtype=np.float64)  # maps current frame -> frame 0
    transforms = [transform.copy()]          # transform actually used per output frame

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count - 1), desc="Stabilization"):
        ret, frame = cap.read()
        if not ret:
            break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cur_pts is not None and len(cur_pts) >= 8:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, cur_gray, cur_pts, None, **LK_PARAMS)
            if new_pts is not None:
                ok = status.ravel() == 1
                new_pts, kept_anchors = new_pts[ok], anchors[ok]

                if len(new_pts) >= 8:
                    m, inliers = cv2.findHomography(
                        new_pts, kept_anchors, cv2.RANSAC, RANSAC_THRESH)
                    if m is not None:
                        transform = m / m[2, 2]
                        # keep inliers only: person / bad tracks fall out
                        inliers = inliers.ravel().astype(bool)
                        new_pts = new_pts[inliers]
                        kept_anchors = kept_anchors[inliers]

                cur_pts, anchors = new_pts, kept_anchors

        # on any failure above, the previous transform is reused as-is
        stabilized = cv2.warpPerspective(frame, transform, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        out.write(stabilized)
        transforms.append(transform.copy())

        # replenish the pool with fresh corners anchored via the current fit
        if cur_pts is None or len(cur_pts) < MIN_POINTS:
            fresh = detect_corners(cur_gray, existing=cur_pts)
            if fresh is not None and len(fresh) > 0:
                ones = np.ones((len(fresh), 1, 1))
                fresh_h = np.concatenate([fresh.astype(np.float64), ones],
                                         axis=2)
                fresh_anchors = (transform @ fresh_h.reshape(-1, 3).T).T
                fresh_anchors = fresh_anchors[:, :2] / fresh_anchors[:, 2:3]
                fresh_anchors = fresh_anchors.reshape(-1, 1, 2).astype(np.float32)
                if cur_pts is None or len(cur_pts) == 0:
                    cur_pts, anchors = fresh, fresh_anchors
                else:
                    cur_pts = np.concatenate([cur_pts, fresh])
                    anchors = np.concatenate([anchors, fresh_anchors])

        prev_gray = cur_gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return np.array(transforms, dtype=np.float64)