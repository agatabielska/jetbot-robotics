"""Dual-head PilotNet: classifier head for `left` (7 bins) + regressor head for `forward`.

Forward output is a TUPLE (left_logits (B, 7), forward_pred (B, 1)) in training.
For ONNX export, we wrap this into a single (B, 2) output via argmax + bin_centers,
producing the standard forward_left contract expected by bot_driving.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


N_LEFT_BINS = 7


def left_bin_centers() -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, N_LEFT_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2.0


class _PilotBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, 5, stride=2), nn.BatchNorm2d(36), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, 5, stride=2), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        with torch.no_grad():
            self.flat_dim = self.conv(torch.zeros(1, 3, 224, 224)).numel()
        self.shared = nn.Sequential(
            nn.Linear(self.flat_dim, 100), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.shared(x)


class Model(nn.Module):
    """
    Training mode: forward returns (left_logits, forward_pred) — TUPLE.
    Export mode (`export=True`): forward returns (B, 2) tensor — converted via argmax → bin_center.
    """
    INPUT_SIZE = 224
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        self.backbone = _PilotBackbone()
        self.head_left = nn.Linear(50, N_LEFT_BINS)
        self.head_forward = nn.Sequential(nn.Linear(50, 10), nn.ReLU(inplace=True),
                                          nn.Linear(10, 1), nn.Tanh())
        self.register_buffer(
            "_centers", torch.tensor(left_bin_centers(), dtype=torch.float32)
        )
        self.export_mode = False

    def forward(self, x):
        z = self.backbone(x)
        left_logits = self.head_left(z)
        forward_pred = self.head_forward(z)
        if self.export_mode:
            # softmax-weighted bin center (smoother than hard argmax for ONNX trace)
            left_soft = torch.softmax(left_logits, dim=-1)
            left_val = (left_soft * self._centers.unsqueeze(0)).sum(dim=-1, keepdim=True)
            # Clip slightly inside (-1, 1) for strict assert
            left_val = torch.clamp(left_val, -0.999, 0.999)
            return torch.cat([forward_pred, left_val], dim=-1)
        return left_logits, forward_pred
