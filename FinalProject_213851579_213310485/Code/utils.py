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



