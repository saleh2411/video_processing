
import json
import os
import time

from utils import path_from_root
from stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import matting
from tracking import tracking

ID1 = '213851579'
ID2 = '213310485'


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

    # Step 1: Stabilization
    print('Step 1: Stabilization')
    transforms = stabilize_video(input_video, stabilize_path)
    timing['stabilize'] = time.time() - start_time

    # Step 2: Background Subtraction
    print('Step 2: Background Subtraction')
    background_subtraction(stabilize_path, extracted_path, binary_path, transforms)
    timing['extracted'] = time.time() - start_time
    timing['binary'] = time.time() - start_time

    # Step 3: Matting
    print('Step 3: Matting')
    matting(stabilize_path, binary_path, background_image, matted_path, alpha_path)
    timing['alpha'] = time.time() - start_time
    timing['matted'] = time.time() - start_time

    # Step 4: Tracking
    print('Step 4: Tracking')
    tracking_json_path = os.path.join(outputs_dir, 'tracking.json')
    tracking(matted_path, binary_path, output_path, tracking_json_path)
    timing['output'] = time.time() - start_time

    timing_path = os.path.join(outputs_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f)



if __name__ == '__main__':
    main()
