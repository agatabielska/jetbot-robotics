"""Train PilotNet on JetBot recordings.

Loss = 1.0 * MSE(steer) + 0.3 * MSE(throttle).
best.pt is checkpointed on val_steer_mse (steering > throttle in importance).

Iteration 2 changes:
- WeightedRandomSampler with inverse-frequency weights on 10 steering bins —
  fights the right-turn bias that hflip alone couldn't fix.
- LR warmup (3 epochs linear 0 -> args.lr) then cosine for the rest —
  breaks the long "predict the mean" plateau seen in iteration 1.
- 50 epochs default (was 30) since warmup eats 3 and we want more time
  past plateau-break.
- Dumps hflip debug pairs to <out_dir>/debug_hflip/ on startup.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import JetBotDataset, _debug_dump_batch
from model import PilotNet
from split import build_splits


def _epoch(model, loader, optimizer, scaler, device, train: bool, steer_w=1.0, throt_w=0.3):
    model.train(train)
    sum_steer, sum_throt, n = 0.0, 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                preds = model(imgs)
                steer_mse = nn.functional.mse_loss(preds[:, 0], targets[:, 0])
                throt_mse = nn.functional.mse_loss(preds[:, 1], targets[:, 1])
                loss = steer_w * steer_mse + throt_w * throt_mse
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        bs = imgs.size(0)
        sum_steer += steer_mse.item() * bs
        sum_throt += throt_mse.item() * bs
        n += bs
    return sum_steer / n, sum_throt / n


def _build_steering_sampler(train_samples, n_bins: int = 10, smoothing: int = 50):
    """Inverse-frequency weights on n_bins steering bins, with smoothing
    so very rare bins don't dominate to the point of overfitting."""
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    steers = np.array([s[1] for s in train_samples])
    bin_idx = np.clip(np.digitize(steers, edges[1:-1]), 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    bin_weights = 1.0 / (bin_counts + smoothing)
    sample_weights = bin_weights[bin_idx]
    print(f"  steering bin counts: {bin_counts.tolist()}")
    print(f"  steering bin weights: {[f'{w:.3e}' for w in bin_weights]}")
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_samples),
        replacement=True,
    )


def _build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / max(warmup_epochs, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1)
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/dataset")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="pilotnet/runs/exp1")
    ap.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--color_jitter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--filter_reverse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--steering_sampler", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 3
        args.warmup_epochs = 1
        args.batch_size = 32

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_samples, val_samples = build_splits(
        args.data_dir, seed=args.seed, filter_reverse=args.filter_reverse
    )

    _debug_dump_batch(
        [s for s in train_samples if abs(s[1]) > 0.5],
        os.path.join(args.out_dir, "debug_hflip"),
        n=4,
        seed=args.seed,
    )

    train_ds = JetBotDataset(
        train_samples,
        train=True,
        color_jitter=args.color_jitter,
        hflip=args.hflip,
        seed=args.seed,
    )
    val_ds = JetBotDataset(val_samples, train=False)

    if args.steering_sampler:
        sampler = _build_steering_sampler(train_samples)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PilotNet().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = _build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(device="cuda") if device.type == "cuda" else None

    best_val_steer = float("inf")
    history = []
    log_path = os.path.join(args.out_dir, "log.jsonl")
    log_f = open(log_path, "w")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_s, tr_t = _epoch(model, train_loader, optimizer, scaler, device, train=True)
        va_s, va_t = _epoch(model, val_loader, optimizer, scaler, device, train=False)
        scheduler.step()
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_steer_mse": tr_s,
            "train_throttle_mse": tr_t,
            "train_total": tr_s + 0.3 * tr_t,
            "val_steer_mse": va_s,
            "val_throttle_mse": va_t,
            "val_total": va_s + 0.3 * va_t,
            "lr": optimizer.param_groups[0]["lr"],
            "dt_s": dt,
        }
        history.append(row)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        improved = va_s < best_val_steer
        if improved:
            best_val_steer = va_s
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        print(
            f"ep {epoch:2d}/{args.epochs}  "
            f"tr_s={tr_s:.4f}  tr_t={tr_t:.4f}  "
            f"va_s={va_s:.4f}  va_t={va_t:.4f}  "
            f"lr={row['lr']:.2e}  {dt:.1f}s"
            + ("  *" if improved else "")
        )
    log_f.close()
    print(f"done. best val_steer_mse = {best_val_steer:.4f}")


if __name__ == "__main__":
    main()
