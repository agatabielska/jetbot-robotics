"""
test_onnx.py  —  Run ONNX models on random images from the dataset and compare
                 predictions vs ground truth labels.

Usage:
    # Test a single model on 10 random images:
    python test_onnx.py --onnx experiments/tiny_96/jetbot_tiny_96.onnx --dataset dataset

    # Test all models at once and compare:
    python test_onnx.py --scan-dir experiments --dataset dataset --n 20
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import glob

# ── Preprocessing (must match training) ──────────────────────────────────────
IMG_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def postprocess(onnx_output) -> tuple:
    raw = np.array(onnx_output[0]).flatten()
    raw = np.clip(raw, -1.0, 1.0)
    return float(raw[0]), float(raw[1])


# ── Load dataset samples (image path + ground truth) ─────────────────────────

def load_all_samples(dataset_dir: str) -> list:
    """Returns list of (img_path, gt_forward, gt_left)."""
    samples = []
    for csv_path in sorted(glob.glob(os.path.join(dataset_dir, "*.csv"))):
        session = os.path.splitext(os.path.basename(csv_path))[0]
        img_dir = os.path.join(dataset_dir, session)
        df = pd.read_csv(csv_path, header=None, names=["frame_id", "forward", "left"])
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, f"{int(row.frame_id):04d}.jpg")
            if os.path.exists(img_path):
                samples.append((img_path, float(row.forward), float(row.left)))
    return samples


# ── Run test for one model ────────────────────────────────────────────────────

def test_model(onnx_path: str, samples: list, n: int):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    picked = random.sample(samples, min(n, len(samples)))

    errors_fwd, errors_left = [], []

    print(f"\n{'─'*70}")
    print(f"Model: {onnx_path}")
    print(f"{'Image':<45} {'GT fwd':>7} {'GT left':>8} {'Pr fwd':>7} {'Pr left':>8} {'Δfwd':>6} {'Δleft':>7}")
    print("─" * 70)

    for img_path, gt_fwd, gt_left in picked:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Could not read {img_path}")
            continue

        inp = preprocess(frame)
        out = sess.run(None, {"input": inp})
        pr_fwd, pr_left = postprocess(out)

        e_fwd  = abs(pr_fwd  - gt_fwd)
        e_left = abs(pr_left - gt_left)
        errors_fwd.append(e_fwd)
        errors_left.append(e_left)

        short_name = os.path.join(
            os.path.basename(os.path.dirname(img_path)),
            os.path.basename(img_path)
        )
        print(f"  {short_name:<43} {gt_fwd:>7.3f} {gt_left:>8.3f} "
              f"{pr_fwd:>7.3f} {pr_left:>8.3f} "
              f"{e_fwd:>6.3f} {e_left:>7.3f}")

    if errors_fwd:
        print("─" * 70)
        print(f"  {'MAE':.<43} {np.mean(errors_fwd):>7.3f} {np.mean(errors_left):>8.3f}")
        print(f"  {'Max error':.<43} {np.max(errors_fwd):>7.3f} {np.max(errors_left):>8.3f}")
        within_01_fwd  = np.mean(np.array(errors_fwd)  < 0.1)
        within_01_left = np.mean(np.array(errors_left) < 0.1)
        print(f"  {'Within ±0.1':.<43} {within_01_fwd:>7.1%} {within_01_left:>8.1%}")

    return np.mean(errors_fwd) if errors_fwd else None, \
           np.mean(errors_left) if errors_left else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx",      help="Path to a single .onnx file")
    group.add_argument("--scan-dir",  help="Scan folder recursively for *.onnx files")
    p.add_argument("--dataset", required=True, help="Path to dataset/ folder")
    p.add_argument("--n",       type=int, default=10, help="Number of random images to test")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading dataset samples from: {args.dataset}")
    samples = load_all_samples(args.dataset)
    print(f"Found {len(samples)} labelled images total.")

    if not samples:
        print("No samples found — check your dataset path.")
        sys.exit(1)

    onnx_files = []
    if args.onnx:
        onnx_files = [args.onnx]
    else:
        for root, _, files in os.walk(args.scan_dir):
            for f in files:
                if f.endswith(".onnx"):
                    onnx_files.append(os.path.join(root, f))

    if not onnx_files:
        print("No .onnx files found.")
        sys.exit(1)

    summary = []
    for onnx_path in sorted(onnx_files):
        mae_fwd, mae_left = test_model(onnx_path, samples, args.n)
        summary.append((onnx_path, mae_fwd, mae_left))

    if len(summary) > 1:
        print(f"\n\n{'='*70}")
        print("Model comparison (MAE on same random sample)")
        print("─" * 70)
        print(f"  {'Model':<45} {'MAE fwd':>8} {'MAE left':>9}")
        print("─" * 70)
        for path, mf, ml in summary:
            name = os.path.basename(path)
            print(f"  {name:<45} {mf:>8.3f} {ml:>9.3f}")


if __name__ == "__main__":
    main()