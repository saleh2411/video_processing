import os
import cv2
import numpy as np

#from claude
def project_root():
    """Absolute path of the project root (the parent of the Code folder)."""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(code_dir, os.pardir))

#from claude
def path_from_root(*parts):
    return os.path.join(project_root(), *parts)

#new added
def open_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    return video

#from ex1
def get_video_parameters(capture):
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: VideoCapture object. The input video's VideoCapture.
    Returns:
        parameters: dict. A dictionary of parameters names to their values.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    parameters = {"fourcc": fourcc, "fps": fps, "height": height, "width": width}
    return parameters


#new, inspired by ex2
def create_video(video_path, fps, w, h):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h), isColor=True)
    if not out.isOpened():
        raise ValueError(f"Could not create video {video_path}")
    return out


#new added for matting/tracking
def frames_of(video_path):
    """Generator over the BGR frames of a video file."""
    video = open_video(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame
    video.release()


#new added for matting/tracking
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



