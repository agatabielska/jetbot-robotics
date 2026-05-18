"""Hyperparameter / architecture tuner wrapper for `dualhead.py`.

This script runs `dualhead.py` for a set of experiments (either provided as
a JSON list of param dicts or generated from a simple grid) and collects
outputs (stdout/stderr), plus any best-metadata JSON saved by the training
script. Results are saved under an output folder for later analysis.

Usage examples:
  # simple grid (generate combinations)
  python utils/hypersearch.py --out results/hypersearch_1 \
    --grid '{"dropout": [0.3, 0.5], "lr": [1e-4, 1e-3], "freeze_epochs": [0,5]}'

  # explicit experiments file (JSON list of dicts)
  python utils/hypersearch.py --out results/hypersearch_2 --experiments experiments.json

Notes:
  - This wrapper runs experiments sequentially. For parallel execution,
    run multiple instances or extend this script.
  - It expects to be executed in the `/home/agata/Documents/studia/6sem/robotics/model`
    workspace (so that `python3 dualhead.py` resolves correctly).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List


def cartesian_from_grid(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [list(grid[k]) for k in keys]
    experiments: List[Dict[str, Any]] = []
    for combo in product(*values):
        exp = {k: v for k, v in zip(keys, combo)}
        experiments.append(exp)
    return experiments


def build_cmd(exp: Dict[str, Any], results_root: Path, idx: int, base_name: str = 'run') -> List[str]:
    """Convert an experiment dict into a `dualhead.py` CLI invocation.

    Keys are mapped to CLI flags by replacing underscores with hyphens
    and prefixing with `--`. Boolean True values emit the flag, False
    values are omitted.
    """
    cmd = ["python3", "dualhead.py"]
    # ensure each run writes into its own onnx run folder under results_root
    run_name = f"{base_name}_{idx:03d}"
    cmd += ["--export-onnx", "--onnx-output-root", str(results_root), "--run-name", run_name]

    for k, v in exp.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    return cmd


def find_best_metadata(results_root: Path, run_name: str) -> Dict[str, Any] | None:
    # possible filenames observed: best_metadata.json, best_epoch_metadata.json
    run_dir = results_root / run_name
    if not run_dir.exists():
        return None
    for fname in run_dir.iterdir():
        if fname.suffix == '.json' and 'best' in fname.stem:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                    return {'file': str(fname), 'payload': payload}
            except Exception:
                continue
    # fallback: check for bestmodels/<run_name>/ etc
    alt = Path('bestmodels') / run_name
    if alt.exists():
        for fname in alt.iterdir():
            if fname.suffix == '.json' and 'best' in fname.stem:
                try:
                    with open(fname, 'r', encoding='utf-8') as f:
                        payload = json.load(f)
                        return {'file': str(fname), 'payload': payload}
                except Exception:
                    continue
    return None


def run_experiments(experiments: List[Dict[str, Any]], out_dir: Path, sequential: bool = True) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    results: List[Dict[str, Any]] = []

    for i, exp in enumerate(experiments, start=1):
        run_name = f"run_{i:03d}"
        cmd = build_cmd(exp, out_dir, i, base_name='run')
        cmd_str = ' '.join(shlex.quote(c) for c in cmd)
        print(f"[{i}/{len(experiments)}] Running: {cmd_str}")

        t0 = datetime.utcnow().isoformat()
        start_ts = time.time()

        proc = subprocess.run(cmd, cwd='.', capture_output=True, text=True)

        duration = time.time() - start_ts
        t1 = datetime.utcnow().isoformat()

        # write logs
        stdout_path = logs_dir / f"{run_name}.stdout.txt"
        stderr_path = logs_dir / f"{run_name}.stderr.txt"
        with open(stdout_path, 'w', encoding='utf-8') as f:
            f.write(proc.stdout)
        with open(stderr_path, 'w', encoding='utf-8') as f:
            f.write(proc.stderr)

        # collect metadata if any
        found = find_best_metadata(out_dir, f"run{i:03d}")

        result = {
            'index': i,
            'run_name': f"run{i:03d}",
            'cmd': cmd,
            'cmd_str': cmd_str,
            'start': t0,
            'end': t1,
            'duration_seconds': duration,
            'returncode': proc.returncode,
            'stdout': str(stdout_path),
            'stderr': str(stderr_path),
            'best_metadata': found,
            'params': exp,
        }

        results.append(result)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='Output folder for results (e.g. results/hypersearch_1)')
    p.add_argument('--experiments', help='JSON file with list of experiment dicts')
    p.add_argument('--grid', help='JSON object representing a grid (keys -> list of values)')
    p.add_argument('--max-experiments', type=int, default=0, help='Limit number of experiments (0 = all)')
    args = p.parse_args()

    out_dir = Path(args.out)

    experiments: List[Dict[str, Any]] = []
    if args.experiments:
        with open(args.experiments, 'r', encoding='utf-8') as f:
            experiments = json.load(f)
            if not isinstance(experiments, list):
                raise SystemExit('experiments file must be a JSON list of dicts')
    elif args.grid:
        grid = json.loads(args.grid)
        experiments = cartesian_from_grid(grid)
    else:
        raise SystemExit('Provide --experiments or --grid')

    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]

    print(f'Preparing {len(experiments)} experiments; results -> {out_dir}')
    results = run_experiments(experiments, out_dir)

    summary = {
        'created_at': datetime.utcnow().isoformat(),
        'n_experiments': len(results),
        'results': results,
    }
    summary_path = out_dir / 'experiments_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Finished. Summary written to {summary_path}')


if __name__ == '__main__':
    main()
