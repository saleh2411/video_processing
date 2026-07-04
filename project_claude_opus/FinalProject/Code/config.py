"""Central configuration: student IDs, relative paths and per-stage constants.

Every tunable number in the pipeline lives here so the algorithms stay readable
and so a single edit re-tunes a stage.  All paths are built with ``os.path`` and
are relative to this file, so the project runs unchanged from any working
directory (required by the brief: no absolute paths).
"""
import os

# --- student IDs (used in every output file name) ---------------------------
ID1 = "213851579"
ID2 = "213310485"

# --- folder layout ----------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)               # the FinalProject folder


def _find_input_dir():
    """Brief calls the folder 'Input'; the supplied template ships 'Inputs'.

    Accept either so the code is correct against both the spec and the files we
    were actually given.
    """
    for name in ("Input", "Inputs"):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d):
            return d
    return os.path.join(ROOT, "Input")


INPUT_DIR = _find_input_dir()
OUTPUT_DIR = os.path.join(ROOT, "Outputs")
TEMP_DIR = os.path.join(ROOT, "Temp")

INPUT_VIDEO = os.path.join(INPUT_DIR, "INPUT.avi")
BACKGROUND_IMAGE = os.path.join(INPUT_DIR, "background.jpg")


def out_path(stage):
    """Absolute path of an output video, e.g. stabilize -> Outputs/stabilize_ID1_ID2.avi."""
    return os.path.join(OUTPUT_DIR, f"{stage}_{ID1}_{ID2}.avi")


# ---------------------------------------------------------------------------
# Stabilization (Lucas-Kanade global-translation estimate + warp)
# ---------------------------------------------------------------------------
STAB_PROC_WIDTH = 480      # motion is estimated on frames downscaled to this width
STAB_NUM_LEVELS = 4        # pyramid depth for the coarse-to-fine LK solve
STAB_MAX_ITER = 8          # LK refinement iterations per pyramid level
STAB_SMOOTH_RADIUS = 0     # 0 = lock to average position (static camera); >0 = smooth-follow (keeps a slow pan)

# ---------------------------------------------------------------------------
# Background subtraction (median background model + colour distance)
# ---------------------------------------------------------------------------
BGS_MEDIAN_SAMPLES = 40    # frames sampled evenly to build the median/spread background
BGS_Z_THRESH = 4.5         # Mahalanobis z-score above which a pixel is foreground
BGS_OPEN_KERNEL = 7        # opening kernel: removes speckle and thin false streaks
BGS_CLOSE_KERNEL = 15      # closing kernel: fills holes inside the person
BGS_MIN_AREA_FRAC = 0.003  # blobs smaller than this fraction of the frame are dropped
BGS_BORDER_BLACK = 12      # pixels darker than this (all channels) = stabilization border

# ---------------------------------------------------------------------------
# Matting (trimap + distance-transform alpha)
# ---------------------------------------------------------------------------
MAT_FG_ERODE = 7           # erode mask -> "definitely foreground"
MAT_BG_DILATE = 15         # dilate mask -> complement is "definitely background"
MAT_ALPHA_BLUR = 3         # gaussian sigma that softens the final alpha edge

# ---------------------------------------------------------------------------
# Tracking (colour particle filter)
# ---------------------------------------------------------------------------
TRK_NUM_PARTICLES = 150
TRK_POS_NOISE = 6          # std of the position drift noise (px)
TRK_VEL_NOISE = 1          # std of the velocity drift noise (px/frame)
TRK_SMOOTH = 0.6           # EMA weight of the new measurement (1 = no smoothing)
TRK_BOX_COLOR = (0, 255, 0)
TRK_BOX_THICKNESS = 3
# tracking.json key of the first frame. 0 => keys are 0..N-1 (frame index).
TRK_KEY_OFFSET = 0
