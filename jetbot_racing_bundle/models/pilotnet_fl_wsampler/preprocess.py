"""Per-model preprocess for deployment. Single-source-of-truth from Phase 2."""


import numpy as np


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> (1, 3, 224, 224) float32, /255."""
    # Inlined here so the model folder is fully self-contained for Docker.
    import cv2
    if img_bgr.shape[:2] != (224, 224):
        img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    arr = img_bgr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)
