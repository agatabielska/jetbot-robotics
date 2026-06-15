"""Interactive JetBot calibration: tune differential.{left,right} so the robot drives straight.

How to use:
  1) Place JetBot on a flat surface with ~1m of clear space ahead.
  2) Run: python3 calibrate.py --config /workspace/config.yml
  3) Use keyboard:
       w  -> drive forward at constant speed (forward=0.5, left=0)
       s  -> stop
       q  -> decrease differential.left  (slows left wheel — robot will veer LEFT less)
       e  -> increase differential.left
       a  -> decrease differential.right
       d  -> increase differential.right
       r  -> reset to (1.00, 1.00)
       p  -> save current values to config.yml
       x  -> quit
  4) Sequence: press `w` (drive), observe drift, stop with `s`, adjust with q/e/a/d, repeat.
  5) When the robot drives straight at `w` for ~1 meter, press `p` to save, then `x` to quit.

Note: this script reuses `jetbot` library and is intended to run INSIDE the Docker container
on the actual robot. On a development host it'll exit with a clear message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

STEP = 0.05
FORWARD_SPEED = 0.5


def _safe_import_jetbot():
    try:
        from jetbot import Robot
        return Robot
    except Exception as e:
        print(f"[calibrate] Cannot import `jetbot` — are you on the actual JetBot? ({e})")
        sys.exit(2)


def _safe_import_sshkeyboard():
    try:
        from sshkeyboard import listen_keyboard, stop_listening
        return listen_keyboard, stop_listening
    except Exception:
        print("[calibrate] Install sshkeyboard: `pip3 install sshkeyboard`")
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/workspace/config.yml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {"robot": {"max_speed": 0.22, "max_steering": 0.5, "differential": {"left": 1.0, "right": 1.0}}}

    Robot = _safe_import_jetbot()
    listen_keyboard, stop_listening = _safe_import_sshkeyboard()
    robot = Robot()
    state = {
        "diff_left": float(config["robot"]["differential"]["left"]),
        "diff_right": float(config["robot"]["differential"]["right"]),
        "max_speed": float(config["robot"]["max_speed"]),
        "driving": False,
    }

    def _apply(driving: bool):
        f = FORWARD_SPEED if driving else 0.0
        l = f * state["diff_left"] * state["max_speed"]
        r = f * state["diff_right"] * state["max_speed"]
        robot.set_motors(left_speed=l, right_speed=r)
        print(f"  diff=({state['diff_left']:.2f}, {state['diff_right']:.2f})  motors=({l:+.3f}, {r:+.3f})  driving={driving}")

    def _save():
        config["robot"]["differential"]["left"] = round(state["diff_left"], 4)
        config["robot"]["differential"]["right"] = round(state["diff_right"], 4)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(config, open(cfg_path, "w"))
        print(f"[calibrate] Saved to {cfg_path}: differential=({state['diff_left']:.2f}, {state['diff_right']:.2f})")

    def on_key(key):
        if key == "w":
            state["driving"] = True; _apply(True)
        elif key == "s":
            state["driving"] = False; _apply(False)
        elif key == "q":
            state["diff_left"] = max(0.0, state["diff_left"] - STEP); _apply(state["driving"])
        elif key == "e":
            state["diff_left"] = min(1.2, state["diff_left"] + STEP); _apply(state["driving"])
        elif key == "a":
            state["diff_right"] = max(0.0, state["diff_right"] - STEP); _apply(state["driving"])
        elif key == "d":
            state["diff_right"] = min(1.2, state["diff_right"] + STEP); _apply(state["driving"])
        elif key == "r":
            state["diff_left"] = 1.0; state["diff_right"] = 1.0; _apply(state["driving"])
        elif key == "p":
            _save()
        elif key == "x":
            robot.set_motors(0.0, 0.0); stop_listening()

    print("[calibrate] Keys: w=drive, s=stop, q/e=left-, a/d=right±, r=reset, p=save, x=quit")
    print(f"[calibrate] Current: diff=({state['diff_left']:.2f}, {state['diff_right']:.2f}), max_speed={state['max_speed']:.2f}")
    listen_keyboard(on_press=on_key, sequential=True, until=None)
    robot.set_motors(0.0, 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
