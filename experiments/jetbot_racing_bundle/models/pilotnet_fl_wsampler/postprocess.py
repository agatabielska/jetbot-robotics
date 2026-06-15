"""Per-model postprocess for forward_left target. Strict-bounds clip to (-0.999, 0.999)."""


import numpy as np

CLIP = np.float32(0.999)


def postprocess(detections: np.ndarray) -> np.ndarray:
    """ONNX output -> (2,) float32 (forward, left), clipped strictly inside (-1, 1)."""
    arr = np.asarray(detections, dtype=np.float32).reshape(-1)
    out = np.array([arr[0], arr[1]], dtype=np.float32)
    return np.clip(out, -CLIP, CLIP).astype(np.float32)
