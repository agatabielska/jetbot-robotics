"""
JetBot Model Architectures
Three options ranked by speed vs accuracy tradeoff:
  1. TinyCNN      – fastest, ~200k params, good baseline
  2. MobileNetV2  – best accuracy, ~2.2M params, pretrained
  3. ShuffleNetV2 – middle ground, ~350k params, pretrained

All output 2 values in [-1, 1] via tanh.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── 1. Tiny custom CNN ────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """
    ~200k parameters. Runs at ~3-5ms on Jetson Nano at 96×96.
    """

    def __init__(self, img_size: int = 96, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,   16, stride=2),   # 96→48
            self._block(16,  32, stride=2),   # 48→24
            self._block(32,  64, stride=2),   # 24→12
            self._block(64, 128, stride=2),   # 12→6
            self._block(128, 128, stride=2),  # 6→3
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


# ── 2. MobileNetV2 (pretrained) ───────────────────────────────────────────────

class MobileNetV2Driver(nn.Module):
    """
    ~2.2M params total, ~600k in the fine-tuned layers.
    Pretrained on ImageNet – great feature extractor for RGB road images.
    """

    def __init__(self, dropout: float = 0.3, freeze_until: int = 14):
        """
        freeze_until: freeze the first N feature layers (out of 19).
                      14 freezes most of the backbone; set 0 to train all.
        """
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze early layers
        features = list(base.features.children())
        for layer in features[:freeze_until]:
            for p in layer.parameters():
                p.requires_grad = False

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


# ── 3. ShuffleNetV2 ×0.5 (pretrained) ────────────────────────────────────────

class ShuffleNetDriver(nn.Module):
    """
    ~350k params. Faster than MobileNetV2, better than TinyCNN.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        base = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        )
        # Remove original classifier
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(name: str = "tiny", **kwargs) -> nn.Module:
    """
    name: "tiny" | "mobilenet" | "shufflenet"
    """
    registry = {
        "tiny":       TinyCNN,
        "mobilenet":  MobileNetV2Driver,
        "shufflenet": ShuffleNetDriver,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ("tiny", "mobilenet", "shufflenet"):
        m = build_model(name)
        dummy = torch.randn(2, 3, 96, 96)
        out = m(dummy)
        print(f"{name:12s}  params={count_params(m):>8,}  output={out.shape}  range=[{out.min():.2f}, {out.max():.2f}]")