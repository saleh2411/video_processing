"""Basic Video Processing methods."""
import os
import cv2


ID1 = '213851579'
ID2 = '213310485'

INPUT_VIDEO = 'atrium.avi'
GRAYSCALE_VIDEO = f'{ID1}_{ID2}_atrium_grayscale.avi'
BLACK_AND_WHITE_VIDEO = f'{ID1}_{ID2}_atrium_black_and_white.avi'
SOBEL_VIDEO = f'{ID1}_{ID2}_atrium_sobel.avi'


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


def convert_video_to_grayscale(input_video_path, output_video_path):
    """Convert the video in the input path to grayscale.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, convert it to gray scale,
    and save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    writer = cv2.VideoWriter(
        output_video_path,
        params["fourcc"],
        params["fps"],
        (params["width"], params["height"]),
        isColor=False,
    )
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        writer.write(gray)
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


def convert_video_to_black_and_white(input_video_path, output_video_path):
    """Convert the video in the input path to black and white.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    writer = cv2.VideoWriter(
        output_video_path,
        params["fourcc"],
        params["fps"],
        (params["width"], params["height"]),
        isColor=True,
    )
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, bw = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
        bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        writer.write(bw_rgb)
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


def convert_video_to_sobel(input_video_path, output_video_path):
    """Convert the video in the input path to sobel map.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    capture = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(capture)
    writer = cv2.VideoWriter(
        output_video_path,
        params["fourcc"],
        params["fps"],
        (params["width"], params["height"]),
        isColor=False,
    )
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, ddepth=-1, dx=1, dy=1, ksize=5)
        writer.write(sobel)
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


def main():
    convert_video_to_grayscale(INPUT_VIDEO, GRAYSCALE_VIDEO)
    convert_video_to_black_and_white(INPUT_VIDEO, BLACK_AND_WHITE_VIDEO)
    convert_video_to_sobel(INPUT_VIDEO, SOBEL_VIDEO)


if __name__ == "__main__":
    main()
