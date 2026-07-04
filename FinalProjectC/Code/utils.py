"""Shared I/O helpers for the final project pipeline.

All paths in the project are built relative to the project root using os.path,
so the code runs correctly with `python Code/main.py` from any working directory.
"""
import os

import cv2
import numpy as np


def project_root():
    """Absolute path of the project root (the parent of the Code folder)."""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(code_dir, os.pardir))


def path_from_root(*parts):
    return os.path.join(project_root(), *parts)


def open_video(path):
    """Open a video for reading and return (capture, params dict)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError('Could not open video: {}'.format(path))
    params = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, params


def create_writer(path, fps, width, height):
    """Create a color XVID .avi writer."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise IOError('Could not open video writer: {}'.format(path))
    return writer


def frames_of(path):
    """Generator over the BGR frames of a video file."""
    cap, _ = open_video(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def to_color(gray_or_mask):
    """Replicate a single-channel image to 3 channels (all outputs are color)."""
    return cv2.merge([gray_or_mask, gray_or_mask, gray_or_mask])


def largest_component(mask):
    """Keep only the largest connected component of a 0/1 uint8 mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def fill_holes(mask):
    """Fill internal holes of a 0/1 uint8 mask via flood fill from the border."""
    h, w = mask.shape
    inv = (1 - mask).astype(np.uint8)
    padded = np.pad(inv, 1, mode='constant', constant_values=1)
    flood = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 0)
    holes = padded[1:-1, 1:-1]
    return np.clip(mask + holes, 0, 1).astype(np.uint8)


def bbox_of_mask(mask):
    """Bounding box (row, col, height, width) of the nonzero part of a mask,
    or None if the mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)
