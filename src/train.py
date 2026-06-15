"""
JetBot Training Script
───────────────────────
Trains a regression model to predict (forward, left) from camera images.

Usage:
    python train.py --dataset ./dataset --model tiny --img-size 96 --epochs 30
    python train.py --dataset ./dataset --model mobilenet --img-size 96 --epochs 40 --lr 3e-4
    python train.py --dataset ./dataset --model shufflenet --img-size 96 --epochs 35

After training, a .onnx file is exported automatically.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jetbot_dataset import build_dataloaders
from jetbot_models import build_model, count_params


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray, tol: float = 0.1):
    """
    preds, targets: shape (N, 2)  columns: [forward, left]

    Returns dict with:
      mae_fwd, mae_left  – mean absolute error per output
      r2_fwd, r2_left    – R² (coefficient of determination)
      acc_fwd, acc_left  – fraction of samples within `tol`
    """
    errors = np.abs(preds - targets)   # (N, 2)

    mae_fwd  = errors[:, 0].mean()
    mae_left = errors[:, 1].mean()

    def r2(p, t):
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + 1e-8)

    r2_fwd  = r2(preds[:, 0], targets[:, 0])
    r2_left = r2(preds[:, 1], targets[:, 1])

    acc_fwd  = (errors[:, 0] < tol).mean()
    acc_left = (errors[:, 1] < tol).mean()

    return dict(
        mae_fwd=mae_fwd, mae_left=mae_left,
        r2_fwd=r2_fwd,   r2_left=r2_left,
        acc_fwd=acc_fwd, acc_left=acc_left,
    )


# ── Training / eval loops ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        total_loss += loss.item() * len(imgs)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    return total_loss / len(loader.dataset), metrics


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, img_size: int, out_path: str, device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch"}},
        dynamo=False
    )
    print(f"✓ ONNX model exported → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="dataset",     help="Path to dataset/ folder")
    p.add_argument("--model",         default="tiny",        choices=["tiny", "mobilenet", "shufflenet"])
    p.add_argument("--img-size",      type=int, default=96)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--val-split",     type=float, default=0.15)
    p.add_argument("--test-split",    type=float, default=0.1)
    p.add_argument("--future-offset", type=int,   default=0,
                   help="Predict control signals N frames ahead (latency compensation)")
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--out-dir",       default="checkpoints")
    p.add_argument("--loss",          default="huber", choices=["mse", "huber", "mae"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=args.dataset,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        future_offset=args.future_offset,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, dropout=args.dropout).to(device)
    print(f"Model: {args.model}  |  trainable params: {count_params(model):,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Huber is recommended: less sensitive to outlier frames than MSE,
    # but still differentiable unlike pure MAE.
    loss_fn = {
        "mse":   nn.MSELoss(),
        "huber": nn.HuberLoss(delta=0.5),
        "mae":   nn.L1Loss(),
    }[args.loss]

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt = os.path.join(args.out_dir, f"best_{args.model}.pt")

    print(f"\n{'Epoch':>6}  {'Train L':>8}  {'Val L':>8}  "
          f"{'MAE fwd':>8}  {'MAE left':>9}  {'R² fwd':>7}  {'R² left':>8}  "
          f"{'Acc fwd':>8}  {'Acc left':>9}")
    print("─" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, m = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>8.4f}  {val_loss:>8.4f}  "
            f"{m['mae_fwd']:>8.4f}  {m['mae_left']:>9.4f}  "
            f"{m['r2_fwd']:>7.3f}  {m['r2_left']:>8.3f}  "
            f"{m['acc_fwd']:>8.1%}  {m['acc_left']:>9.1%}  "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "metrics": m,
                "args": vars(args),
            }, best_ckpt)
            print(f"           ↑ saved best checkpoint (val_loss={val_loss:.4f})")

    # ── Load best & export ONNX ───────────────────────────────────────────────
    print(f"\nLoading best checkpoint from {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    onnx_path = os.path.join(args.out_dir, f"jetbot_{args.model}_{args.img_size}.onnx")
    export_onnx(model, args.img_size, onnx_path, device)

    print(f"\nDone. Best val loss: {best_val_loss:.4f} at epoch {ckpt['epoch']}")
    print(f"ONNX model: {onnx_path}")
    print("\nUpdate config.yml:")
    print(f"  model:\n    path: '{os.path.abspath(onnx_path)}'")

    # Optional: evaluate on the held-out test set
    if 'test_loader' in locals() and test_loader is not None:
        test_loss, test_metrics = eval_epoch(model, test_loader, loss_fn, device)
        print("\nTest set results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  MAE fwd: {test_metrics['mae_fwd']:.4f}  MAE left: {test_metrics['mae_left']:.4f}")


if __name__ == "__main__":
    main()