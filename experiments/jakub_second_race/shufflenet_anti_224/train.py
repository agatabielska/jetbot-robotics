"""Thin wrapper: calls phase3_common.train.run_from_config."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE3_DIR = Path(__file__).resolve().parents[2]
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))

from phase3_common.train import run_from_config

if __name__ == "__main__":
    result = run_from_config(__file__)
    print(f"\nFINAL: {result}")
