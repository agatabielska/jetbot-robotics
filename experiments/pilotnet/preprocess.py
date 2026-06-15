"""Single source of truth for image preprocessing.

Used by both training (dataset.py) and inference (inference_example.py
plus the on-bot deploy wrapper). Numpy + cv2 only -- NO torch -- so it
runs on the JetBot's onnxruntime stack without extra deps.

Pipeline: BGR -> RGB -> crop top CROP_TOP_FRAC of rows -> resize to
(INPUT_W, INPUT_H) -> /255 -> HWC -> CHW float32.
"""

import cv2
import numpy as np

CROP_TOP_FRAC = 0.40
INPUT_H = 66
INPUT_W = 200


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image (as read by cv2.imread) into the CHW
    float32 tensor the model expects. Shape: (3, INPUT_H, INPUT_W)."""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got {img_bgr.shape}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h = img_rgb.shape[0]
    crop_start = int(round(h * CROP_TOP_FRAC))
    img_cropped = img_rgb[crop_start:, :, :]

    img_resized = cv2.resize(
        img_cropped, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA
    )

    img_float = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    return np.ascontiguousarray(img_chw)
