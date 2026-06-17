"""Train a JetBot line-following end-to-end driving model.

Architecture follows NVIDIA's PilotNet idea but trimmed to ~600k parameters so
it stays cheap enough to run on the JetBot. The model consumes 224x224 BGR
images (channels-first) and outputs (forward, left) signals in (-1, 1).

Outputs:
  - checkpoints/best.pt          : best PyTorch weights (by val loss)
  - checkpoints/jetbot_model.onnx: ONNX export, OPSET 11

Usage:
  python train.py --dataset-root dataset --epochs 40 --batch-size 64
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    path: Path
    forward: float
    left: float


def discover_samples(dataset_root: Path) -> List[Sample]:
    """Find every (image, forward, left) triple in the dataset folder.

    Each recording session has a sibling pair: ``<id>.csv`` and ``<id>/``.
    Only CSV files that have a matching image directory are kept; the
    unrelated ``cars_dataset.csv`` and ``preference.csv`` are skipped.
    """

    samples: List[Sample] = []
    for csv_path in sorted(dataset_root.glob("*.csv")):
        img_dir = dataset_root / csv_path.stem
        if not img_dir.is_dir():
            continue
        df = pd.read_csv(csv_path, header=None, names=["idx", "forward", "left"])
        for _, row in df.iterrows():
            img_path = img_dir / f"{int(row['idx']):04d}.jpg"
            if img_path.is_file():
                samples.append(
                    Sample(
                        path=img_path,
                        forward=float(row["forward"]),
                        left=float(row["left"]),
                    )
                )
    return samples


class JetbotDataset(Dataset):
    """Loads BGR JetBot frames and applies augmentations on the fly.

    Images stay in BGR (uint8) order to match the inference camera pipeline.
    Augmentations focus on illumination changes - as the README recommends -
    plus a horizontal-flip trick that mirrors the ``left`` signal so we
    effectively double the dataset without re-recording.
    """

    def __init__(self, samples: List[Sample], image_size: int = 224, train: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: np.ndarray, left: float) -> Tuple[np.ndarray, float]:
        # Horizontal flip: forward stays, steering sign flips.
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            left = -left

        # Brightness / contrast jitter in HSV space (V channel).
        if random.random() < 0.8:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.7, 1.3)  # saturation
            hsv[..., 2] *= random.uniform(0.6, 1.4)  # value/brightness
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Mild Gaussian noise to make the model robust to sensor noise.
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random shadow / cutout: drop a thin horizontal band of brightness.
        if random.random() < 0.2:
            h = img.shape[0]
            band_h = random.randint(8, 32)
            y0 = random.randint(0, h - band_h)
            img[y0:y0 + band_h] = (img[y0:y0 + band_h] * 0.5).astype(np.uint8)

        return img, left

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(sample.path)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        left = sample.left
        if self.train:
            img, left = self._augment(img, left)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        target = np.array([sample.forward, left], dtype=np.float32)

        return torch.from_numpy(img), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PilotNet(nn.Module):
    """Small PilotNet-inspired regressor, ~660k params at 224x224 input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Output is 128x14x14. We pool to 2x2 (14/2=7 is integer, which
            # is required for ONNX opset 11 to accept adaptive_avg_pool2d).
            nn.AdaptiveAvgPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> Tuple[float, float, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_fwd = 0.0
    total_left = 0.0
    n = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    for imgs, targets in iterator:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            preds = model(imgs)
            loss = F.smooth_l1_loss(preds, targets)
            fwd_err = F.l1_loss(preds[:, 0], targets[:, 0])
            left_err = F.l1_loss(preds[:, 1], targets[:, 1])

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_fwd += fwd_err.item() * bs
        total_left += left_err.item() * bs
        n += bs
        iterator.set_postfix(loss=f"{total_loss / n:.4f}")

    return total_loss / n, total_fwd / n, total_left / n


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(model: nn.Module, out_path: Path, device: torch.device) -> None:
    """Export the model to ONNX with OPSET 11 (required by the JetBot stack)."""

    model.eval().to(device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``dynamo=False`` keeps the legacy TorchScript exporter, which is what
    # honors ``opset_version=11`` (required by the JetBot's onnxruntime).
    torch.onnx.export(
        model,
        (dummy,),
        out_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    # Sanity check: load with onnxruntime and verify numerical match.
    import onnxruntime as rt

    sess = rt.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    np_input = dummy.detach().cpu().numpy().astype(np.float32)
    onnx_out = sess.run(None, {"input": np_input})[0]
    with torch.no_grad():
        torch_out = model(dummy).detach().cpu().numpy()
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    print(f"ONNX vs PyTorch max abs diff: {max_diff:.2e}")
    print(f"ONNX saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    samples = discover_samples(args.dataset_root)
    if not samples:
        raise SystemExit(f"No samples found under {args.dataset_root}")
    print(f"Discovered {len(samples)} samples")

    # Random train/val split (good enough for this dataset size).
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * args.val_split))
    val_idx = set(indices[:n_val].tolist())

    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = JetbotDataset(train_samples, train=True)
    val_ds = JetbotDataset(val_samples, train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = PilotNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    best_path = args.out_dir / "best.pt"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, t_fwd, t_left = run_epoch(
            model, train_loader, optimizer, device, f"train {epoch}/{args.epochs}"
        )
        val_loss, v_fwd, v_left = run_epoch(
            model, val_loader, None, device, f"  val {epoch}/{args.epochs}"
        )
        scheduler.step()

        print(
            f"epoch {epoch:02d} | "
            f"train {train_loss:.4f} (fwd {t_fwd:.3f}, left {t_left:.3f}) | "
            f"val {val_loss:.4f} (fwd {v_fwd:.3f}, left {v_left:.3f}) | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved new best (val={best_val:.4f}) to {best_path}")

    elapsed = time.time() - start
    print(f"Training done in {elapsed/60:.1f} min. Best val loss: {best_val:.4f}")

    # Reload best weights and export to ONNX (OPSET 11).
    model.load_state_dict(torch.load(best_path, map_location=device))
    onnx_path = args.out_dir / "jetbot_model.onnx"
    export_onnx(model, onnx_path, device)


if __name__ == "__main__":
    main()
