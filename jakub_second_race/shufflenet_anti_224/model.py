"""ShuffleNetV2 x0.5 (pretrained ImageNet) + regression head — 224x224 variant.

Identyczny z shufflenet_anti_96, tylko INPUT_SIZE=224 (bogatsze cechy, wolniejszy inference).
~407k params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class Model(nn.Module):
    INPUT_SIZE = 224
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        backbone = shufflenet_v2_x0_5(weights=weights)
        in_feats = backbone.fc.in_features  # 1024
        backbone.fc = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.OUTPUT_DIM),
            nn.Tanh(),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
