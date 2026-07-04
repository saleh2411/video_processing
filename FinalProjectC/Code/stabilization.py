"""Video stabilization (lectures: motion representation, optical flow, RANSAC).

Method
------
The camera in INPUT.avi is hand-held and shaky while the scene itself is static,
so we stabilize every frame onto the coordinate system of the FIRST frame:

1. Detect corners in the previous frame with Shi-Tomasi `goodFeaturesToTrack`
   (the min-eigenvalue variant of the Harris corner detector).
2. Track them into the current frame with pyramidal Lucas-Kanade optical flow.
3. Estimate the frame-to-frame homography with RANSAC. The walking person
   violates the global background motion model, so the points on him are
   rejected as RANSAC outliers (the static background provides the majority
   of the tracked corners).
4. Compose the per-frame homographies to map the current frame all the way
   back to frame 0, and warp with `warpPerspective`.
5. Drift correction: composing ~200 homographies accumulates small errors,
   so after warping we run a second Lucas-Kanade pass between the reference
   frame and the warped frame and apply the small residual homography.

Black pixels near the borders after warping are expected and allowed (FAQ #1).
No blur / smoothing is applied to the frames themselves so the matting stage
is not harmed.
"""
import cv2
import numpy as np

from utils import open_video, create_writer

# Shi-Tomasi corner detection parameters
FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.01, minDistance=20, blockSize=11)
# Pyramidal Lucas-Kanade parameters
LK_PARAMS = dict(winSize=(21, 21), maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
RANSAC_REPROJ_THRESH = 3.0


def _homography_between(gray_src, gray_dst):
    """Homography mapping points of gray_src onto gray_dst using
    Shi-Tomasi corners + pyramidal LK + RANSAC. Returns None on failure."""
    p0 = cv2.goodFeaturesToTrack(gray_src, mask=None, **FEATURE_PARAMS)
    if p0 is None or len(p0) < 8:
        return None
    p1, status, _ = cv2.calcOpticalFlowPyrLK(gray_src, gray_dst, p0, None, **LK_PARAMS)
    if p1 is None:
        return None
    good = status.ravel() == 1
    src_pts, dst_pts = p0[good], p1[good]
    if len(src_pts) < 8:
        return None
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    return H


def stabilize_video(input_path, output_path):
    """Stabilize input_path onto its first frame and write to output_path."""
    cap, prm = open_video(input_path)
    w, h, fps = prm['width'], prm['height'], prm['fps']
    writer = create_writer(output_path, fps, w, h)

    ok, first = cap.read()
    if not ok:
        raise IOError('Empty input video: {}'.format(input_path))
    writer.write(first)

    ref_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray
    H_cum = np.eye(3, dtype=np.float64)  # maps current frame -> frame 0 coords

    n_done = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # frame-to-frame motion (current -> previous)
        H = _homography_between(gray, prev_gray)
        if H is not None:
            H_cum = H_cum @ H
            H_cum /= H_cum[2, 2]

        stabilized = cv2.warpPerspective(frame, H_cum, (w, h),
                                         flags=cv2.INTER_LINEAR)

        # residual drift correction against the reference frame
        warped_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        H_corr = _homography_between(warped_gray, ref_gray)
        if H_corr is not None:
            H_cum = H_corr @ H_cum
            H_cum /= H_cum[2, 2]
            stabilized = cv2.warpPerspective(frame, H_cum, (w, h),
                                             flags=cv2.INTER_LINEAR)

        writer.write(stabilized)
        prev_gray = gray
        n_done += 1
        if n_done % 50 == 0:
            print('  stabilization: {} frames done'.format(n_done))

    cap.release()
    writer.release()
    print('  stabilization: finished ({} frames)'.format(n_done))
