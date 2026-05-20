"""Quick-setup `differential` calibration for JetBot.

Pre-set values from the PUT course-author table (PUT_jetbot/README.md):
    JetBot 01 -> left=1.00, right=0.90
    JetBot 02 -> left=0.85, right=1.00
    JetBot 03 -> left=1.00, right=1.00
    JetBot 04 -> left=1.00, right=1.00

Usage (interactive):
    python3 setup_differential.py
    # → Enter JetBot number (1-4) or 'c' for custom: 2

Usage (one-liner via CLI flags):
    python3 setup_differential.py --jetbot 2
    python3 setup_differential.py --left 0.92 --right 1.0
    python3 setup_differential.py --output /workspace/config.yml --jetbot 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PRESETS = {
    1: (1.00, 0.90),
    2: (0.85, 1.00),
    3: (1.00, 1.00),
    4: (1.00, 1.00),
}

DEFAULT_OUTPUT = Path("/workspace/config.yml")


def _load(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {
        "model": {"path": ""},
        "robot": {"max_speed": 0.22, "max_steering": 0.5,
                  "differential": {"left": 1.0, "right": 1.0}},
    }


def _save(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[setup_differential] Saved to: {path}")
    print(yaml.safe_dump(config["robot"], sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick differential calibration for JetBot.")
    ap.add_argument("--jetbot", type=int, choices=[1, 2, 3, 4], default=None,
                    help="JetBot number (1-4); applies PUT preset values")
    ap.add_argument("--left", type=float, default=None, help="Custom differential.left")
    ap.add_argument("--right", type=float, default=None, help="Custom differential.right")
    ap.add_argument("--max-speed", type=float, default=None, help="Optional override of max_speed")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="config.yml path")
    args = ap.parse_args()

    # Determine values
    if args.jetbot is not None:
        left, right = PRESETS[args.jetbot]
        source = f"JetBot {args.jetbot} preset (PUT_jetbot/README.md)"
    elif args.left is not None and args.right is not None:
        left, right = args.left, args.right
        source = "CLI --left/--right"
    else:
        # interactive
        print("=== JetBot differential setup ===")
        print("Presets (from PUT_jetbot/README.md):")
        for i, (l, r) in PRESETS.items():
            print(f"  {i} -> left={l:.2f}, right={r:.2f}")
        print("  c -> custom values")
        choice = input("Enter choice (1-4 or c): ").strip().lower()
        if choice in {"1", "2", "3", "4"}:
            left, right = PRESETS[int(choice)]
            source = f"JetBot {choice} preset"
        elif choice == "c":
            left = float(input("differential.left (0.0-1.2): "))
            right = float(input("differential.right (0.0-1.2): "))
            source = "interactive custom"
        else:
            print(f"[setup_differential] Invalid choice: {choice}", file=sys.stderr)
            return 2

    # Update config
    config = _load(args.output)
    config.setdefault("robot", {})
    config["robot"].setdefault("differential", {})
    config["robot"]["differential"]["left"] = round(float(left), 4)
    config["robot"]["differential"]["right"] = round(float(right), 4)
    if args.max_speed is not None:
        config["robot"]["max_speed"] = round(float(args.max_speed), 4)

    print(f"[setup_differential] Source: {source}")
    print(f"[setup_differential] differential.left={left:.2f}  differential.right={right:.2f}")
    _save(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
