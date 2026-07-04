"""Final Project - orchestrator.

Run with:  ``python Code/main.py``

Loads INPUT.avi once, runs the four stages in memory, and writes the six output
videos plus timing.json and tracking.json into the Outputs folder.  Each video's
time is stamped in timing.json right after it is fully written, measured from the
start of this script (as required by the brief).
"""
import gc
import json
import os
import time

import cv2
import numpy as np

import config as cfg
import video_utils as vu
import stabilization
import background_subtraction
import matting
import tracking


def _bbox_from_masks(masks):
    """First non-empty mask's bounding box as [ROW, COL, HEIGHT, WIDTH].

    Falls back to a centered box if every mask is empty.
    """
    for m in masks:
        ys, xs = np.where(m > 0)
        if ys.size > 0:
            row, col = int(ys.min()), int(xs.min())
            return [row, col, int(ys.max() - ys.min() + 1),
                    int(xs.max() - xs.min() + 1)]
    h, w = masks[0].shape[:2]
    return [h // 4, w // 4, h // 2, w // 2]


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    timing = {}

    print("Loading INPUT.avi ...")
    frames, params = vu.load_frames(cfg.INPUT_VIDEO)
    fps = params["fps"]
    print(f"  {len(frames)} frames  {params['width']}x{params['height']}  {fps} fps")

    # 1. Stabilization -----------------------------------------------------
    print("Stage 1/4  stabilization ...")
    stab = stabilization.stabilize(frames, params)
    vu.save_frames(cfg.out_path("stabilize"), stab, fps)
    timing["time_to_stabilize"] = time.time() - t0
    del frames
    gc.collect()

    # 2. Background subtraction -> extracted + binary ----------------------
    print("Stage 2/4  background subtraction ...")
    extracted, masks = background_subtraction.subtract(stab, params)
    vu.save_frames(cfg.out_path("extracted"), extracted, fps)
    vu.save_binary(cfg.out_path("binary"), masks, fps)
    timing["time_to_binary"] = time.time() - t0
    del extracted
    gc.collect()

    # 3. Matting -> alpha + matted -----------------------------------------
    print("Stage 3/4  matting ...")
    bg_image = cv2.imread(cfg.BACKGROUND_IMAGE)
    matted, alpha = matting.matte(stab, masks, bg_image, params)
    vu.save_alpha(cfg.out_path("alpha"), alpha, fps)
    timing["time_to_alpha"] = time.time() - t0
    vu.save_frames(cfg.out_path("matted"), matted, fps)
    timing["time_to_matted"] = time.time() - t0
    init_box = _bbox_from_masks(masks)
    del stab, alpha
    gc.collect()

    # 4. Tracking -> OUTPUT + tracking.json --------------------------------
    print("Stage 4/4  tracking ...")
    output, boxes = tracking.track(matted, masks, params, init_box)
    del masks
    gc.collect()
    vu.save_frames(cfg.out_path("OUTPUT"), output, fps)
    timing["time_to_output"] = time.time() - t0

    # JSON artifacts -------------------------------------------------------
    with open(os.path.join(cfg.OUTPUT_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=4)
    with open(os.path.join(cfg.OUTPUT_DIR, "tracking.json"), "w") as f:
        json.dump(boxes, f, indent=4)

    print("Done. Outputs written to", cfg.OUTPUT_DIR)
    for k, v in timing.items():
        print(f"  {k}: {v:.1f}s")


if __name__ == "__main__":
    main()
