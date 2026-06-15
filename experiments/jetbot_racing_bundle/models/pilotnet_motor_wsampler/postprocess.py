"""Postprocess for motor_lr target: convert (m_l, m_r) -> (forward, left), then clip."""


import numpy as np

CLIP = np.float32(0.999)
MAX_STEERING = 0.5  # must match common.targets.forward_left_to_motors_raw default


def _motors_to_forward_left(m_l: float, m_r: float):
    if m_l < m_r:
        forward = m_r
        left = (m_r - m_l) / MAX_STEERING
    elif m_l > m_r:
        forward = m_l
        left = -(m_l - m_r) / MAX_STEERING
    else:
        forward = m_l
        left = 0.0
    return forward, left


def postprocess(detections: np.ndarray) -> np.ndarray:
    arr = np.asarray(detections, dtype=np.float32).reshape(-1)
    m_l, m_r = float(arr[0]), float(arr[1])
    forward, left = _motors_to_forward_left(m_l, m_r)
    out = np.array([forward, left], dtype=np.float32)
    return np.clip(out, -CLIP, CLIP).astype(np.float32)
