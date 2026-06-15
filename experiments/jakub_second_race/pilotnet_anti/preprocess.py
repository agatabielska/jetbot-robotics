"""BGR /255 preprocess for PilotNet (input 224x224, custom train-from-scratch)."""

import numpy as np

INPUT_SIZE = 224


def preprocess(img_bgr):
    """BGR uint8 HWC -> (1, 3, 224, 224) float32, /255."""
    import cv2
    if img_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img_bgr = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = img_bgr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)
