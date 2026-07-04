"""Tracking (lecture: Kalman filter).

The walking person is selected AUTOMATICALLY in the matted video: the matted
frames are, by construction, equal to the new background wherever there is no
person, so thresholding the difference between a matted frame and the (resized)
background image detects the person without any manual initialization.

The per-frame detection (bounding box of the largest difference component) is
the MEASUREMENT of a Kalman filter, as taught in the lecture:

  state     x = [row_c, col_c, height, width, v_row, v_col]^T
  dynamics  constant-velocity for the box center, random-walk for the size
  measure   z = [row_c, col_c, height, width]^T

  predict:  x = F x,  P = F P F^T + Q
  update:   K = P H^T (H P H^T + R)^(-1),
            x = x + K (z - H x),  P = (I - K H) P

The filter smooths the raw detections and keeps predicting the box on frames
where the detection is missing. The smoothed box is drawn (green rectangle) on
every matted frame -> OUTPUT video, and written per frame to tracking.json as
[ROW, COL, HEIGHT, WIDTH] (top-left row/col), with keys "1".."N".
"""
import json

import cv2
import numpy as np

from utils import open_video, create_writer, frames_of, largest_component, \
    fill_holes, bbox_of_mask

DIFF_THRESH = 30   # |matted - background| threshold for person detection
OPEN_RADIUS = 5
CLOSE_RADIUS = 15


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over [row_c, col_c, h, w, v_row, v_col]."""

    def __init__(self, z0):
        self.x = np.array([z0[0], z0[1], z0[2], z0[3], 0.0, 0.0], np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        self.F = np.eye(6)
        self.F[0, 4] = 1.0  # row_c += v_row
        self.F[1, 5] = 1.0  # col_c += v_col
        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5, 4.0, 4.0])
        self.R = np.diag([25.0, 25.0, 50.0, 50.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.asarray(z, np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x


def _detect_person(matted_frame, background):
    """Measurement [row_c, col_c, h, w] from |matted - background|, or None."""
    diff = np.abs(matted_frame.astype(np.int16) - background.astype(np.int16))
    mask = (diff.max(axis=2) > DIFF_THRESH).astype(np.uint8)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_RADIUS, OPEN_RADIUS))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_RADIUS, CLOSE_RADIUS))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    box = bbox_of_mask(mask)
    if box is None:
        return None
    r, c, bh, bw = box
    return np.array([r + bh / 2.0, c + bw / 2.0, bh, bw], np.float64)


def tracking(matted_path, background_image_path, output_path, tracking_json_path):
    """Write the OUTPUT video (green rectangle around the tracked person) and
    tracking.json. Returns the wall-clock time at which OUTPUT finished."""
    import time

    cap, prm = open_video(matted_path)
    w, h, fps = prm['width'], prm['height'], prm['fps']
    cap.release()

    background = cv2.imread(background_image_path)
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)

    writer = create_writer(output_path, fps, w, h)
    tracker = None
    results = {}

    n_done = 0
    for frame in frames_of(matted_path):
        z = _detect_person(frame, background)
        if tracker is None:
            if z is not None:
                tracker = KalmanBoxTracker(z)
            state = z if z is not None else np.array([h / 2.0, w / 2.0, h / 2.0, w / 4.0])
        else:
            tracker.predict()
            if z is not None:
                tracker.update(z)
            state = tracker.x

        rc, cc, bh, bw = state[0], state[1], state[2], state[3]
        row = int(round(max(0, min(rc - bh / 2.0, h - 1))))
        col = int(round(max(0, min(cc - bw / 2.0, w - 1))))
        bh = int(round(min(bh, h - row)))
        bw = int(round(min(bw, w - col)))

        n_done += 1
        results[str(n_done)] = [row, col, bh, bw]
        cv2.rectangle(frame, (col, row), (col + bw, row + bh), (0, 255, 0), 3)
        writer.write(frame)
        if n_done % 50 == 0:
            print('  tracking: {} frames done'.format(n_done))

    writer.release()
    output_done_time = time.time()

    with open(tracking_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print('  tracking: finished ({} frames)'.format(n_done))
    return output_done_time
