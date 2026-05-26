"""RGB + ImageNet normalization for pretrained ShuffleNetV2 (input 96x96)."""

import numpy as np

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 96


def preprocess(img_bgr):
    """BGR uint8 HWC -> (1, 3, 96, 96) float32, RGB ImageNet-normalized."""
    import cv2
    if img_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img_bgr = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)
