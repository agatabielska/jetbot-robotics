"""
CV-based JetBot training script.
Trains a small MLP: (cx_norm, cy_norm) → (forward, turn).

Usage:
    python cv_train.py
    python cv_train.py --model mlp_deep --epochs 100 --lr 3e-4
    python cv_train.py --dataset ../dataset_preprocessed/dataset/train --out-dir cv_checkpoints

After training the best checkpoint is exported as an ONNX model.
To run on the JetBot:
    1. Apply binary threshold to each camera frame → detect road centre (cx, cy)
    2. Normalise: cx_norm = cx / 224,  cy_norm = cy / 224
    3. Feed [cx_norm, cy_norm] to the ONNX model
    4. Read output [forward, turn] and pass to PUTDriver.update()
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cv_dataset import build_dataloaders
from cv_models import IMG_SIZE, build_model, count_params


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray, tol: float = 0.1) -> dict:
    """
    preds, targets: shape (N, 2)  columns: [forward, turn]
    Returns mae, R², and within-tolerance accuracy for each output.
    """
    errors = np.abs(preds - targets)

    mae_fwd  = errors[:, 0].mean()
    mae_turn = errors[:, 1].mean()

    def r2(p, t):
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + 1e-8)

    r2_fwd  = r2(preds[:, 0], targets[:, 0])
    r2_turn = r2(preds[:, 1], targets[:, 1])

    acc_fwd  = (errors[:, 0] < tol).mean()
    acc_turn = (errors[:, 1] < tol).mean()

    return dict(
        mae_fwd=mae_fwd,  mae_turn=mae_turn,
        r2_fwd=r2_fwd,    r2_turn=r2_turn,
        acc_fwd=acc_fwd,  acc_turn=acc_turn,
    )


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for coords, labels in loader:
        coords, labels = coords.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(coords)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(coords)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for coords, labels in loader:
        coords, labels = coords.to(device), labels.to(device)
        preds = model(coords)
        loss  = criterion(preds, labels)
        total_loss += loss.item() * len(coords)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    return total_loss / len(loader.dataset), metrics


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, out_path: str, device):
    model.eval()
    dummy = torch.zeros(1, 2, device=device)   # [cx_norm, cy_norm]
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=11,
        input_names=["coords"],
        output_names=["output"],
        dynamic_axes={"coords": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"ONNX model exported → {out_path}  ({size_kb:.1f} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    here    = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(here, "..", "dataset_preprocessed", "dataset", "train")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",      default=default_dataset,
                   help="Path to dataset_preprocessed/dataset/train/ (default: auto-detected)")
    p.add_argument("--model",        default="mlp", choices=["mlp", "mlp_deep"])
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--val-split",    type=float, default=0.15)
    p.add_argument("--loss",         default="huber", choices=["mse", "huber", "mae"])
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--out-dir",      default=os.path.join(here, "cv_checkpoints"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        root_dir=args.dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, dropout=args.dropout).to(device)
    print(f"Model    : {args.model}  |  trainable params: {count_params(model):,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = {
        "mse":   nn.MSELoss(),
        "huber": nn.HuberLoss(delta=0.5),
        "mae":   nn.L1Loss(),
    }[args.loss]

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt = os.path.join(args.out_dir, f"best_{args.model}.pt")

    print(
        f"\n{'Epoch':>6}  {'Train L':>8}  {'Val L':>8}  "
        f"{'MAE fwd':>8}  {'MAE turn':>9}  {'R² fwd':>7}  {'R² turn':>8}  "
        f"{'Acc fwd':>8}  {'Acc turn':>9}"
    )
    print("─" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, m = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"{epoch:>6}  {train_loss:>8.4f}  {val_loss:>8.4f}  "
            f"{m['mae_fwd']:>8.4f}  {m['mae_turn']:>9.4f}  "
            f"{m['r2_fwd']:>7.3f}  {m['r2_turn']:>8.3f}  "
            f"{m['acc_fwd']:>8.1%}  {m['acc_turn']:>9.1%}  "
            f"({time.time() - t0:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "metrics":     m,
                "args":        vars(args),
                "img_size":    IMG_SIZE,
            }, best_ckpt)
            print(f"           ↑ saved best checkpoint (val_loss={val_loss:.4f})")

    # ── Load best & export ONNX ───────────────────────────────────────────────
    print(f"\nLoading best checkpoint: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    onnx_path = os.path.join(args.out_dir, f"jetbot_cv_{args.model}.onnx")
    export_onnx(model, onnx_path, device)

    print(f"\nDone. Best val loss: {best_val_loss:.4f} at epoch {ckpt['epoch']}")
    print(f"ONNX: {onnx_path}")
    print(f"\nInference snippet for bot_driving.py:")
    print(f"  session = ort.InferenceSession('{onnx_path}')")
    print(f"  coords  = np.array([[cx / {IMG_SIZE}, cy / {IMG_SIZE}]], dtype=np.float32)")
    print(f"  forward, turn = session.run(None, {{'coords': coords}})[0][0]")


if __name__ == "__main__":
    main()
