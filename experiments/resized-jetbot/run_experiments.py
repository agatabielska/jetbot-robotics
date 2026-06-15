"""
JetBot Multi-Model Experiment Runner
──────────────────────────────────────
Trains all three model variants and produces a comparison table.
Useful before class to pick the best model.

Usage:
    python run_experiments.py --dataset ./dataset
"""

import argparse
import subprocess
import sys
import json
import os

EXPERIMENTS = [
    # (name,          model,        img_size, epochs, lr,    future_offset)
    ("tiny_96",       "tiny",       96,       35,     1e-3,  0),
    ("tiny_96_f2",    "tiny",       96,       35,     1e-3,  2),   # 2-frame lookahead
    ("shuffle_96",    "shufflenet", 96,       35,     5e-4,  0),
    ("mobilenet_96",  "mobilenet",  96,       40,     3e-4,  0),
]


def run_experiment(dataset, exp, out_dir):
    name, model, img_size, epochs, lr, future_offset = exp
    exp_out = os.path.join(out_dir, name)
    os.makedirs(exp_out, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--dataset",       dataset,
        "--model",         model,
        "--img-size",      str(img_size),
        "--epochs",        str(epochs),
        "--lr",            str(lr),
        "--future-offset", str(future_offset),
        "--out-dir",       exp_out,
        "--loss",          "huber",
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"  model={model}, img_size={img_size}, epochs={epochs}, "
          f"lr={lr}, future_offset={future_offset}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--only",    default=None, help="Run only this experiment name")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        name = exp[0]
        if args.only and name != args.only:
            continue
        success = run_experiment(args.dataset, exp, args.out_dir)
        results.append((name, "✓" if success else "✗"))

    print("\n\nExperiment Summary")
    print("─" * 30)
    for name, status in results:
        print(f"  {status}  {name}")

    print("\nONNX files:")
    for root, _, files in os.walk(args.out_dir):
        for f in files:
            if f.endswith(".onnx"):
                print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    main()