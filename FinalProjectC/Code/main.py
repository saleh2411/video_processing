"""Video Processing final project - main entry point.

Run from the project root with:  python Code/main.py

Pipeline (flow diagram of the project PDF):
  INPUT.avi -> Stabilization -> Background Subtraction -> Matting -> Tracking
producing in the Outputs folder:
  stabilize_ID1_ID2.avi, extracted_ID1_ID2.avi, binary_ID1_ID2.avi,
  alpha_ID1_ID2.avi, matted_ID1_ID2.avi, OUTPUT_ID1_ID2.avi,
  timing.json, tracking.json

All paths are relative, built with os.path from the location of this file, so
the script works no matter what the current working directory is.
"""
import json
import os
import time

from utils import path_from_root
from stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import matting
from tracking import tracking

ID1 = '213310485'
ID2 = '213851579'


def main():
    start_time = time.time()

    input_video = path_from_root('Input', 'INPUT.avi')
    background_image = path_from_root('Input', 'background.jpg')
    outputs_dir = path_from_root('Outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    def out(name):
        return os.path.join(outputs_dir, '{}_{}_{}.avi'.format(name, ID1, ID2))

    stabilize_path = out('stabilize')
    extracted_path = out('extracted')
    binary_path = out('binary')
    alpha_path = out('alpha')
    matted_path = out('matted')
    output_path = out('OUTPUT')

    timing = {}

    print('[1/4] Stabilization')
    stabilize_video(input_video, stabilize_path)
    timing['time_to_stabilize'] = round(time.time() - start_time, 2)

    print('[2/4] Background subtraction')
    binary_done = background_subtraction(stabilize_path, binary_path, extracted_path)
    timing['time_to_binary'] = round(binary_done - start_time, 2)

    print('[3/4] Matting')
    alpha_done, matted_done = matting(stabilize_path, binary_path,
                                      background_image, matted_path, alpha_path)
    timing['time_to_alpha'] = round(alpha_done - start_time, 2)
    timing['time_to_matted'] = round(matted_done - start_time, 2)

    print('[4/4] Tracking')
    output_done = tracking(matted_path, background_image, output_path,
                           os.path.join(outputs_dir, 'tracking.json'))
    timing['time_to_output'] = round(output_done - start_time, 2)

    with open(os.path.join(outputs_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f, indent=4)

    print('Done. Total runtime: {:.1f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
