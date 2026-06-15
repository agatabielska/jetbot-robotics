"""MobileNetV3-Small (torchvision pretrained ImageNet) + regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class Model(nn.Module):
    INPUT_SIZE = 224
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        in_feats = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_feats, 256), nn.Hardswish(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, self.OUTPUT_DIM), nn.Tanh(),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
