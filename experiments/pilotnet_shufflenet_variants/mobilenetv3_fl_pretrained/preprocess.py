"""RGB + ImageNet normalization for pretrained MobileNet."""


import numpy as np

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> (1, 3, 224, 224) float32, RGB ImageNet-normalized."""
    import cv2
    if img_bgr.shape[:2] != (224, 224):
        img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)
