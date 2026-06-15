"""Canonical PilotNet (NVIDIA end-to-end driving paper).

Input: (N, 3, 66, 200) float32 in [0, 1].
Output: (N, 2) = [steering, throttle].
"""

import torch
import torch.nn as nn

from preprocess import INPUT_H, INPUT_W


class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
            flat_dim = self.features(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 2),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"PilotNet params: {total:,}  (flatten dim = {flat_dim})")

    def forward(self, x):
        return torch.tanh(self.head(self.features(x)))


if __name__ == "__main__":
    m = PilotNet()
    out = m(torch.zeros(2, 3, INPUT_H, INPUT_W))
    print(f"output shape: {tuple(out.shape)}")
