import os
import cv2

import os

import cv2
import numpy as np


def project_root():
    """Absolute path of the project root (the parent of the Code folder)."""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(code_dir, os.pardir))


def path_from_root(*parts):
    return os.path.join(project_root(), *parts)
