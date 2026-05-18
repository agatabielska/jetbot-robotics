"""
cv_test_onnx.py  —  Evaluate CV ONNX model(s) against preprocessed dataset labels.

For each sample the model receives (cx_norm, cy_norm) and must predict
(forward, turn).  Ground truth comes from the augmented CSV files in
dataset_preprocessed/.

Usage:
    # Test a single model on 20 random samples from the test split:
    python cv_test_onnx.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx

    # Run against the full test split (no --n cap):
    python cv_test_onnx.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx --all

    # Compare every .onnx in cv_checkpoints/ on 30 random samples:
    python cv_test_onnx.py --scan-dir cv_checkpoints --n 30

    # Use the train split instead of test:
    python cv_test_onnx.py --onnx cv_checkpoints/jetbot_cv_mlp.onnx --split train
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd

IMG_SIZE = 224   # must match cv_models.py


# ── Load samples from preprocessed CSVs ──────────────────────────────────────

def load_samples(dataset_preprocessed_dir: str, split: str) -> list:
    """
    Returns list of (session, frame_id, cx_norm, cy_norm, gt_forward, gt_turn).
    Rows where cx == -1 (no road centre detected) are skipped.
    """
    split_dir = os.path.join(dataset_preprocessed_dir, "dataset", split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    samples = []
    csv_files = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {split_dir}")

    for csv_path in csv_files:
        session = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["frame_id", "forward", "turn", "cx", "cy"],
        )
        df = df[df["cx"] != -1]
        for _, row in df.iterrows():
            samples.append((
                session,
                int(row["frame_id"]),
                float(row["cx"]) / IMG_SIZE,
                float(row["cy"]) / IMG_SIZE,
                float(row["forward"]),
                float(row["turn"]),
            ))

    return samples


# ── Run inference + evaluation for one model ──────────────────────────────────

def test_model(onnx_path: str, samples: list, n: int | None) -> tuple:
    """
    n=None → evaluate all samples.
    Returns (mae_fwd, mae_turn) or (None, None) if no samples evaluated.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    picked = samples if n is None else random.sample(samples, min(n, len(samples)))

    errors_fwd, errors_turn = [], []

    print(f"\n{'─'*80}")
    print(f"Model : {onnx_path}")
    print(f"Tested: {len(picked)} samples")
    print(
        f"{'Session / frame':<38} {'cx_n':>5} {'cy_n':>5}  "
        f"{'GT fwd':>6} {'GT trn':>7}  "
        f"{'Pr fwd':>6} {'Pr trn':>7}  "
        f"{'Δfwd':>5} {'Δtrn':>5}"
    )
    print("─" * 80)

    for session, frame_id, cx_n, cy_n, gt_fwd, gt_turn in picked:
        coords = np.array([[cx_n, cy_n]], dtype=np.float32)
        raw    = sess.run(None, {input_name: coords})[0].flatten()
        pr_fwd, pr_turn = float(np.clip(raw[0], -1, 1)), float(np.clip(raw[1], -1, 1))

        e_fwd  = abs(pr_fwd  - gt_fwd)
        e_turn = abs(pr_turn - gt_turn)
        errors_fwd.append(e_fwd)
        errors_turn.append(e_turn)

        label = f"{session[-10:]}/{frame_id:04d}"
        print(
            f"  {label:<36} {cx_n:>5.3f} {cy_n:>5.3f}  "
            f"{gt_fwd:>6.3f} {gt_turn:>7.3f}  "
            f"{pr_fwd:>6.3f} {pr_turn:>7.3f}  "
            f"{e_fwd:>5.3f} {e_turn:>5.3f}"
        )

    if not errors_fwd:
        print("  (no samples evaluated)")
        return None, None

    arr_fwd  = np.array(errors_fwd)
    arr_turn = np.array(errors_turn)

    print("─" * 80)
    print(f"  {'MAE':<36} {'':>11}  {'':>14}  {arr_fwd.mean():>6.3f} {arr_turn.mean():>7.3f}")
    print(f"  {'Max error':<36} {'':>11}  {'':>14}  {arr_fwd.max():>6.3f} {arr_turn.max():>7.3f}")
    print(
        f"  {'Within ±0.1':<36} {'':>11}  {'':>14}  "
        f"{(arr_fwd < 0.1).mean():>6.1%} {(arr_turn < 0.1).mean():>7.1%}"
    )
    print(
        f"  {'Within ±0.2':<36} {'':>11}  {'':>14}  "
        f"{(arr_fwd < 0.2).mean():>6.1%} {(arr_turn < 0.2).mean():>7.1%}"
    )

    return float(arr_fwd.mean()), float(arr_turn.mean())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here     = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    default_preprocessed = os.path.join(repo_root, "dataset_preprocessed")
    default_scan = os.path.join(here, "cv_checkpoints")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx",     help="Path to a single .onnx file")
    group.add_argument("--scan-dir", nargs="?", const=default_scan,
                       help=f"Scan folder recursively for *.onnx files (default: {default_scan})")
    p.add_argument("--preprocessed", default=default_preprocessed,
                   help="Path to dataset_preprocessed root (default: auto-detected)")
    p.add_argument("--split",  default="test", choices=["train", "test"],
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--n",    type=int, default=20,
                   help="Number of random samples to test (default: 20; ignored with --all)")
    p.add_argument("--all",  action="store_true",
                   help="Evaluate on all samples in the split")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading '{args.split}' samples from: {args.preprocessed}")
    samples = load_samples(args.preprocessed, args.split)
    print(f"Found {len(samples)} valid labelled samples.")

    if not samples:
        print("No samples found — check --preprocessed path.")
        sys.exit(1)

    n = None if args.all else args.n

    # Collect ONNX files
    if args.onnx:
        onnx_files = [args.onnx]
    else:
        scan_dir = args.scan_dir
        onnx_files = sorted(
            path
            for root, _, files in os.walk(scan_dir)
            for f in files
            if f.endswith(".onnx")
            for path in [os.path.join(root, f)]
        )

    if not onnx_files:
        print("No .onnx files found.")
        sys.exit(1)

    summary = []
    for onnx_path in onnx_files:
        mae_fwd, mae_turn = test_model(onnx_path, samples, n)
        summary.append((onnx_path, mae_fwd, mae_turn))

    if len(summary) > 1:
        print(f"\n\n{'='*80}")
        print("Model comparison (MAE on same random sample)")
        print("─" * 80)
        print(f"  {'Model':<50} {'MAE fwd':>8} {'MAE turn':>9}")
        print("─" * 80)
        for path, mf, mt in summary:
            name = os.path.basename(path)
            mf_s = f"{mf:.3f}" if mf is not None else "  N/A "
            mt_s = f"{mt:.3f}" if mt is not None else "  N/A  "
            print(f"  {name:<50} {mf_s:>8} {mt_s:>9}")


if __name__ == "__main__":
    main()
