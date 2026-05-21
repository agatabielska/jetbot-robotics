"""
cv_test_images.py  —  End-to-end evaluation of the CV model on raw images.

Mirrors the full on-robot pipeline for every test frame:
  raw image → binary threshold → detect road centre → normalise → ONNX → predict

This gives results that are directly comparable to resized-jetbot/test_onnx.py
because both scripts start from the same raw images and ground-truth labels.
Frames where the CV pipeline detects no road centre are counted as failures
and excluded from MAE (but reported separately).

Usage:
    python cv_test_images.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx
    python cv_test_images.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx --all
    python cv_test_images.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx --split train --n 50
    python cv_test_images.py --scan-dir cv_checkpoints --n 30
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

# ── CV pipeline constants (must match preprocess_dataset.py) ─────────────────
IMG_SIZE        = 224
THRESHOLD_VALUE = 110
MAX_VALUE       = 255


def apply_binary_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

    h, w = binary.shape
    binary[: (h * 2) // 3, :] = 0
    binary[:, : w // 3]        = 0
    binary[:, (w * 2) // 3 :]  = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


def detect_centre(binary: np.ndarray):
    """Return (cx, cy) of the largest white component, or None."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas))
    return int(round(centroids[best + 1][0])), int(round(centroids[best + 1][1]))


# ── Load samples from the original (unprocessed) dataset ─────────────────────

def load_samples(dataset_dir: str) -> list:
    """Returns list of (img_path, gt_forward, gt_turn)."""
    samples = []
    for csv_path in sorted(glob.glob(os.path.join(dataset_dir, "*.csv"))):
        session = os.path.splitext(os.path.basename(csv_path))[0]
        img_dir = os.path.join(dataset_dir, session)
        df = pd.read_csv(csv_path, header=None, names=["frame_id", "forward", "turn"])
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, f"{int(row.frame_id):04d}.jpg")
            if os.path.exists(img_path):
                samples.append((img_path, float(row.forward), float(row.turn)))
    return samples


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_model(onnx_path: str, samples: list, n: int | None) -> tuple:
    import onnxruntime as ort

    sess       = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    picked = samples if n is None else random.sample(samples, min(n, len(samples)))

    errors_fwd, errors_turn = [], []
    n_no_detection = 0

    print(f"\n{'─'*84}")
    print(f"Model : {onnx_path}")
    print(f"Tested: {len(picked)} frames")
    print(
        f"{'Session / frame':<38} {'cx_n':>5} {'cy_n':>5}  "
        f"{'GT fwd':>6} {'GT trn':>7}  "
        f"{'Pr fwd':>6} {'Pr trn':>7}  "
        f"{'Δfwd':>5} {'Δtrn':>5}"
    )
    print("─" * 84)

    for img_path, gt_fwd, gt_turn in picked:
        image = cv2.imread(img_path)
        if image is None:
            continue

        binary = apply_binary_threshold(image)
        centre = detect_centre(binary)

        session_name = os.path.basename(os.path.dirname(img_path))
        frame_name   = os.path.splitext(os.path.basename(img_path))[0]
        label        = f"{session_name[-10:]}/{frame_name}"

        if centre is None:
            n_no_detection += 1
            print(f"  {label:<36} {'—':>5} {'—':>5}  {gt_fwd:>6.3f} {gt_turn:>7.3f}  {'no detection':>16}")
            continue

        cx, cy     = centre
        cx_n, cy_n = cx / IMG_SIZE, cy / IMG_SIZE
        coords     = np.array([[cx_n, cy_n]], dtype=np.float32)
        raw        = sess.run(None, {input_name: coords})[0].flatten()
        pr_fwd     = float(np.clip(raw[0], -1, 1))
        pr_turn    = float(np.clip(raw[1], -1, 1))

        e_fwd  = abs(pr_fwd  - gt_fwd)
        e_turn = abs(pr_turn - gt_turn)
        errors_fwd.append(e_fwd)
        errors_turn.append(e_turn)

        print(
            f"  {label:<36} {cx_n:>5.3f} {cy_n:>5.3f}  "
            f"{gt_fwd:>6.3f} {gt_turn:>7.3f}  "
            f"{pr_fwd:>6.3f} {pr_turn:>7.3f}  "
            f"{e_fwd:>5.3f} {e_turn:>5.3f}"
        )

    if not errors_fwd:
        print("  (no frames with successful detection)")
        return None, None

    arr_fwd  = np.array(errors_fwd)
    arr_turn = np.array(errors_turn)
    n_scored = len(errors_fwd)
    n_total  = len(picked)

    print("─" * 84)
    print(f"  Detection rate: {n_scored}/{n_total} ({n_scored/n_total:.1%})  —  {n_no_detection} frame(s) with no road centre found")
    print(f"  {'MAE':<36} {'':>11}  {'':>14}  {arr_fwd.mean():>6.3f} {arr_turn.mean():>7.3f}")
    print(f"  {'Max error':<36} {'':>11}  {'':>14}  {arr_fwd.max():>6.3f} {arr_turn.max():>7.3f}")
    print(f"  {'Within ±0.1':<36} {'':>11}  {'':>14}  {(arr_fwd < 0.1).mean():>6.1%} {(arr_turn < 0.1).mean():>7.1%}")
    print(f"  {'Within ±0.2':<36} {'':>11}  {'':>14}  {(arr_fwd < 0.2).mean():>6.1%} {(arr_turn < 0.2).mean():>7.1%}")

    return float(arr_fwd.mean()), float(arr_turn.mean())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here      = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    default_dataset  = os.path.join(repo_root, "put_jetbot_dataset", "dataset", "test")
    default_scan     = os.path.join(here, "cv_checkpoints")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx",     help="Path to a single .onnx file")
    group.add_argument("--scan-dir", nargs="?", const=default_scan,
                       help=f"Scan folder for *.onnx files (default: {default_scan})")
    p.add_argument("--dataset", default=default_dataset,
                   help="Path to put_jetbot_dataset/dataset/{split}/ (default: test split)")
    p.add_argument("--split", default=None, choices=["train", "test"],
                   help="Shortcut: set --dataset to the train or test split automatically")
    p.add_argument("--n",    type=int, default=20,
                   help="Number of random frames to test (default: 20; ignored with --all)")
    p.add_argument("--all",  action="store_true",
                   help="Evaluate on every frame in the split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # --split overrides --dataset
    if args.split:
        here      = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.normpath(os.path.join(here, ".."))
        args.dataset = os.path.join(repo_root, "put_jetbot_dataset", "dataset", args.split)

    print(f"Loading samples from: {args.dataset}")
    samples = load_samples(args.dataset)
    print(f"Found {len(samples)} labelled frames.")

    if not samples:
        print("No samples found — check --dataset path.")
        sys.exit(1)

    n = None if args.all else args.n

    if args.onnx:
        onnx_files = [args.onnx]
    else:
        onnx_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(args.scan_dir)
            for f in files
            if f.endswith(".onnx")
        )

    if not onnx_files:
        print("No .onnx files found.")
        sys.exit(1)

    summary = []
    for onnx_path in onnx_files:
        mae_fwd, mae_turn = test_model(onnx_path, samples, n)
        summary.append((onnx_path, mae_fwd, mae_turn))

    if len(summary) > 1:
        print(f"\n\n{'='*84}")
        print("Model comparison (MAE on same random sample)")
        print("─" * 84)
        print(f"  {'Model':<50} {'MAE fwd':>8} {'MAE turn':>9}")
        print("─" * 84)
        for path, mf, mt in summary:
            name = os.path.basename(path)
            mf_s = f"{mf:.3f}" if mf is not None else "  N/A "
            mt_s = f"{mt:.3f}" if mt is not None else "  N/A  "
            print(f"  {name:<50} {mf_s:>8} {mt_s:>9}")


if __name__ == "__main__":
    main()
