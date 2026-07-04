"""Shared video I/O helpers.

The whole pipeline passes frames as in-memory BGR ``uint8`` NumPy arrays and does
every disk write here.  Keeping I/O in one place means the timing stamps in
``main.py`` are accurate (a video's time is stamped right after it is written)
and the videos are never re-decoded between stages.
"""
import cv2
import numpy as np


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Extract fourcc / fps / height / width / frame_count from a capture."""
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def load_frames(path: str):
    """Read every frame of a video into a list of BGR uint8 arrays.

    Returns (frames, params).  ``params['frame_count']`` is corrected to the
    number of frames actually decoded (the container header is sometimes wrong).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    params = get_video_parameters(cap)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    params["frame_count"] = len(frames)
    return frames, params


def _writer(path: str, fps: int, size, is_color: bool = True) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(fps) if fps and fps > 0 else 30
    return cv2.VideoWriter(path, fourcc, fps, size, isColor=is_color)


def save_frames(path: str, frames, fps: int) -> None:
    """Write colour frames to an XVID .avi.  Frames are clipped to uint8."""
    h, w = frames[0].shape[:2]
    out = _writer(path, fps, (w, h), is_color=True)
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        out.write(f)
    out.release()


def save_binary(path: str, masks, fps: int) -> None:
    """Write 0/1 masks as a video.

    Per project FAQ #5, a [0,1] image is stored by scaling to [0,255] uint8.
    The person becomes white (255), the rest black (0).  Saved as a 3-channel
    video for codec safety (the binary video is exempt from the colour rule).
    """
    h, w = masks[0].shape[:2]
    out = _writer(path, fps, (w, h), is_color=True)
    for m in masks:
        frame = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()


def save_alpha(path: str, alphas, fps: int) -> None:
    """Write alpha maps as a video (uint8, 3-channel).

    Accepts either float alpha in [0,1] (scaled x255) or already-quantised uint8
    in [0,255], so callers may keep whichever representation is cheaper.
    """
    h, w = alphas[0].shape[:2]
    out = _writer(path, fps, (w, h), is_color=True)
    for a in alphas:
        if a.dtype == np.uint8:
            frame = a
        else:
            frame = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()
