"""
export_onnx.py  —  Load .pt checkpoints, sanity-test them, export to ONNX.

Usage:
    # Export a single checkpoint explicitly:
    python export_onnx.py --pt experiments/tiny_96/best_tiny.pt --model tiny --img-size 96

    # Auto-scan an experiments folder and export everything found:
    python export_onnx.py --scan-dir experiments
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── make sure jetbot_models is importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetbot_models import build_model, count_params


# ── helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint(pt_path: str, device: torch.device):
    """Try loading with weights_only=False; print a clear error on failure."""
    try:
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  ✗ Could not load {pt_path}: {e}")
        return None


def infer_model_name(pt_path: str) -> str:
    """
    Guess the model name from the checkpoint path.
    best_tiny.pt       → tiny
    best_mobilenet.pt  → mobilenet
    best_shufflenet.pt → shufflenet
    Falls back to 'tiny' if nothing matches.
    """
    name = os.path.basename(pt_path).lower()
    for candidate in ("mobilenet", "shufflenet", "tiny"):
        if candidate in name:
            return candidate
    return "tiny"


def infer_img_size(pt_path: str) -> int:
    """Guess img_size from parent folder name, e.g. tiny_224 → 224."""
    folder = os.path.basename(os.path.dirname(pt_path))
    for part in folder.split("_"):
        if part.isdigit():
            return int(part)
    return 96   # default


def sanity_test(model: nn.Module, img_size: int, device: torch.device) -> bool:
    """
    Run a random batch through the model and check:
      - output shape is (2,) per sample
      - outputs are in [-1, 1]  (tanh)
    Returns True if all checks pass.
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(4, 3, img_size, img_size, device=device)
        out   = model(dummy)          # (4, 2)

    ok = True
    if out.shape != (4, 2):
        print(f"  ✗ Unexpected output shape: {out.shape}  (expected (4, 2))")
        ok = False

    out_np = out.cpu().numpy()
    if out_np.max() > 1.0 or out_np.min() < -1.0:
        print(f"  ✗ Outputs outside [-1, 1]: min={out_np.min():.3f}  max={out_np.max():.3f}")
        ok = False

    if ok:
        print(f"  ✓ Sanity test passed  —  output shape={tuple(out.shape)}  "
              f"range=[{out_np.min():.3f}, {out_np.max():.3f}]")
    return ok


def export_onnx(model: nn.Module, img_size: int, out_path: str, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✓ ONNX exported → {out_path}  ({size_kb:.1f} KB)")


def verify_onnx(onnx_path: str, img_size: int):
    """Quick ONNX runtime check — optional, skipped if onnxruntime not installed."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        out   = sess.run(None, {"input": dummy})[0]
        print(f"  ✓ ONNX runtime check passed  —  output shape={out.shape}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")
    except ImportError:
        print("  (onnxruntime not installed — skipping runtime verification)")
    except Exception as e:
        print(f"  ✗ ONNX runtime check failed: {e}")


# ── per-checkpoint pipeline ───────────────────────────────────────────────────

def process(pt_path: str, model_name: str, img_size: int, device: torch.device):
    print(f"\n{'─'*60}")
    print(f"Checkpoint : {pt_path}")
    print(f"Model      : {model_name}   img_size={img_size}")

    ckpt = load_checkpoint(pt_path, device)
    if ckpt is None:
        return False

    # Print what was saved in the checkpoint
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    metrics  = ckpt.get("metrics", {})
    print(f"  Saved at epoch {epoch},  val_loss={val_loss:.4f}")
    if metrics:
        print(f"  mae_fwd={metrics.get('mae_fwd', float('nan')):.4f}  "
              f"mae_left={metrics.get('mae_left', float('nan')):.4f}  "
              f"r2_fwd={metrics.get('r2_fwd', float('nan')):.3f}  "
              f"r2_left={metrics.get('r2_left', float('nan')):.3f}")

    # Build model and load weights
    try:
        model = build_model(model_name).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Trainable params: {count_params(model):,}")
    except Exception as e:
        print(f"  ✗ Failed to load state dict: {e}")
        return False

    # Sanity test
    if not sanity_test(model, img_size, device):
        return False

    # Export ONNX next to the .pt file
    out_dir  = os.path.dirname(pt_path)
    onnx_name = f"jetbot_{model_name}_{img_size}.onnx"
    onnx_path = os.path.join(out_dir, onnx_name)
    export_onnx(model, img_size, onnx_path, device)

    # Verify with onnxruntime
    verify_onnx(onnx_path, img_size)

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pt",       help="Path to a single .pt checkpoint")
    group.add_argument("--scan-dir", help="Scan this folder recursively for best_*.pt files")
    p.add_argument("--model",    default=None,
                   help="Model name (tiny|mobilenet|shufflenet). "
                        "Auto-detected from filename if omitted.")
    p.add_argument("--img-size", type=int, default=None,
                   help="Image size used during training. "
                        "Auto-detected from folder name if omitted.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.pt:
        model_name = args.model    or infer_model_name(args.pt)
        img_size   = args.img_size or infer_img_size(args.pt)
        success = process(args.pt, model_name, img_size, device)
        sys.exit(0 if success else 1)

    # --scan-dir mode
    found = []
    for root, _, files in os.walk(args.scan_dir):
        for f in files:
            if f.startswith("best_") and f.endswith(".pt"):
                found.append(os.path.join(root, f))

    if not found:
        print(f"No best_*.pt files found under {args.scan_dir}")
        sys.exit(1)

    print(f"Found {len(found)} checkpoint(s):")
    for p in found:
        print(f"  {p}")

    results = []
    for pt_path in found:
        model_name = args.model    or infer_model_name(pt_path)
        img_size   = args.img_size or infer_img_size(pt_path)
        ok = process(pt_path, model_name, img_size, device)
        results.append((pt_path, ok))

    print(f"\n\n{'='*60}")
    print("Summary")
    print("─" * 60)
    for pt_path, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status}  {pt_path}")

    print("\nONNX files written:")
    for root, _, files in os.walk(args.scan_dir):
        for f in files:
            if f.endswith(".onnx"):
                print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    main()