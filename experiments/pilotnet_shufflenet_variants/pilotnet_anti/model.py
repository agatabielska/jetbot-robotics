"""PilotNet (NVIDIA end-to-end, 1604.07316), adapted for 224x224 input.

Identyczny z pilotnet_fl_wsampler/model.py — różnica leży w configu treningu
(label_shift=2 zamiast 0, w pozostałym wsampler+Huber jak baseline).

5 Conv -> 4 Dense, tanh output dla strict (-1, 1).
~260k params (pod budżetem PUT 600k).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    INPUT_SIZE = 224
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),  # MUSI być 3, nie 4! opset 11 nie wspiera non-integer factor
        )
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 3, self.INPUT_SIZE, self.INPUT_SIZE)).numel()
        self.fc = nn.Sequential(
            nn.Linear(d, 100), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 10), nn.ReLU(inplace=True),
            nn.Linear(10, self.OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
