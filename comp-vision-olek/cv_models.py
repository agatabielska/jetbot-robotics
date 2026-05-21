"""
Coordinate-to-control model for CV-based JetBot driving.

Input : 2 normalised floats  (cx/IMG_SIZE, cy/IMG_SIZE)
Output: 2 floats in [-1, 1]  (forward, turn)  via tanh

Two variants ranked by capacity:
  1. CoordMLP   – tiny 3-layer MLP, ~4 k params, plenty for 2-D input
  2. CoordMLPDeep – 4-layer with skip connection, ~16 k params
"""

import torch
import torch.nn as nn


IMG_SIZE = 224   # must match the camera / preprocessing resolution


class CoordMLP(nn.Module):
    """Tiny MLP: 2 → 64 → 64 → 2."""

    def __init__(self, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordMLPDeep(nn.Module):
    """Deeper MLP with a residual skip: 2 → 128 → 128 → 64 → 2."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(2, 128)
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z + self.block1(z)   # residual
        return self.head(z)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(name: str = "mlp", **kwargs) -> nn.Module:
    """
    name: "mlp" | "mlp_deep"
    """
    registry = {
        "mlp":      CoordMLP,
        "mlp_deep": CoordMLPDeep,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ("mlp", "mlp_deep"):
        m = build_model(name)
        dummy = torch.randn(4, 2)   # batch of 4, 2 normalised coords
        out = m(dummy)
        print(
            f"{name:12s}  params={count_params(m):>6,}  "
            f"output={tuple(out.shape)}  "
            f"range=[{out.min():.3f}, {out.max():.3f}]"
        )
