"""ShuffleNetV2 x0.5 (pretrained ImageNet) + regression head.

Mirror działającego shufflenetu drużynowego z first_race_errors, ale z naszym
anti-bias stackiem (hflip+target_swap, WeightedRandomSampler, label_shift+2,
Huber loss) zastosowanym przez phase3_common/train.py.

~407k params (pod budżetem PUT 600k).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class Model(nn.Module):
    INPUT_SIZE = 96
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
            nn.Tanh(),  # strict (-1, 1) open interval
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
