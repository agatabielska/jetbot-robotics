"""Play back a dataset recording with steering overlay (ground truth vs ONNX model).

Green arrow  = labels from the CSV (human driving).
Red arrow    = model prediction from checkpoints/jetbot_model.onnx.

Usage:
  python test_overlay.py
  python test_overlay.py --session 1652875851.3497071
  python test_overlay.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as rt

from train import Sample, discover_samples


def group_by_session(samples: List[Sample]) -> Dict[str, List[Sample]]:
    sessions: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        sessions[s.path.parent.name].append(s)
    for key in sessions:
        sessions[key].sort(key=lambda s: s.path.name)
    return dict(sorted(sessions.items()))


class OnnxDriver:
    """ONNX inference with the same preprocessing as src/bot_driving.py."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.sess = rt.InferenceSession(
            model_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img: np.ndarray) -> Tuple[float, float]:
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        forward, left = out.reshape(-1).astype(np.float32)
        forward = float(np.clip(forward, -0.999, 0.999))
        left = float(np.clip(left, -0.999, 0.999))
        return forward, left


def draw_steering_arrow(
    img: np.ndarray,
    forward: float,
    left: float,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    origin = (w // 2, h - 18)
    scale = min(w, h) * 0.38
    tip = (
        int(origin[0] - left * scale),
        int(origin[1] - forward * scale),
    )
    cv2.arrowedLine(img, origin, tip, color, thickness, tipLength=0.35)
    cv2.putText(
        img,
        label,
        (8, 18 if label.startswith("GT") else 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    gt_fwd: float,
    gt_left: float,
    pred_fwd: float,
    pred_left: float,
    frame_idx: int,
    total: int,
    session: str,
) -> np.ndarray:
    vis = frame.copy()
    draw_steering_arrow(vis, gt_fwd, gt_left, (0, 220, 0), "GT")
    draw_steering_arrow(vis, pred_fwd, pred_left, (0, 0, 255), "Pred", thickness=2)

    err_fwd = abs(pred_fwd - gt_fwd)
    err_left = abs(pred_left - gt_left)
    lines = [
        f"session: {session}",
        f"frame {frame_idx + 1}/{total}",
        f"GT   fwd={gt_fwd:+.3f}  left={gt_left:+.3f}",
        f"Pred fwd={pred_fwd:+.3f}  left={pred_left:+.3f}",
        f"|err| fwd={err_fwd:.3f}  left={err_left:.3f}",
        "q=quit  space=pause  n=next session",
    ]
    y0 = vis.shape[0] - 8
    for i, line in enumerate(reversed(lines)):
        cv2.putText(
            vis,
            line,
            (8, y0 - i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset playback with steering overlay.")
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/jetbot_model.onnx"),
        help="Path to exported ONNX model.",
    )
    p.add_argument(
        "--session",
        type=str,
        default=None,
        help="Recording folder name (CSV stem). Lists available if omitted.",
    )
    p.add_argument("--fps", type=float, default=15.0, help="Playback frames per second.")
    p.add_argument("--list", action="store_true", help="List sessions and exit.")
    return p.parse_args()


def pick_session(sessions: Dict[str, List[Sample]], requested: str | None) -> str:
    names = list(sessions.keys())
    if requested:
        if requested not in sessions:
            print(f"Unknown session '{requested}'. Available:", file=sys.stderr)
            for n in names:
                print(f"  {n} ({len(sessions[n])} frames)", file=sys.stderr)
            raise SystemExit(1)
        return requested
    print("Available sessions:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n} ({len(sessions[n])} frames)")
    if len(names) == 1:
        return names[0]
    choice = input("Session index or name [0]: ").strip() or "0"
    if choice.isdigit() and int(choice) < len(names):
        return names[int(choice)]
    if choice in sessions:
        return choice
    raise SystemExit(f"Invalid session choice: {choice}")


def play_session(
    session: str,
    samples: List[Sample],
    model: OnnxDriver,
    fps: float,
    all_sessions: Dict[str, List[Sample]],
) -> str | None:
    delay_ms = max(1, int(1000 / fps))
    total = len(samples)
    paused = False
    idx = 0

    while idx < total:
        sample = samples[idx]
        frame = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: could not read {sample.path}, skipping.")
            idx += 1
            continue

        pred_fwd, pred_left = model.predict(frame)
        vis = draw_overlay(
            frame,
            sample.forward,
            sample.left,
            pred_fwd,
            pred_left,
            idx,
            total,
            session,
        )
        cv2.imshow("JetBot steering overlay", vis)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("n"):
            names = list(all_sessions.keys())
            next_i = (names.index(session) + 1) % len(names)
            return names[next_i]
        idx += 1

    return "advance"


def main() -> None:
    args = parse_args()

    samples = discover_samples(args.dataset_root)
    if not samples:
        raise SystemExit(f"No samples under {args.dataset_root}")

    sessions = group_by_session(samples)

    if args.list:
        for name, frames in sessions.items():
            print(f"{name}: {len(frames)} frames")
        return

    model = OnnxDriver(args.model)
    session = pick_session(sessions, args.session)

    print(f"Playing session {session} ({len(sessions[session])} frames)")
    print(f"Model: {args.model.resolve()}")

    while session is not None:
        result = play_session(
            session,
            sessions[session],
            model,
            args.fps,
            sessions,
        )
        if result is None:
            break
        if result == "advance":
            names = list(sessions.keys())
            next_i = (names.index(session) + 1) % len(names)
            session = names[next_i]
            print(f"\nNext session: {session}")
            time.sleep(0.3)
        else:
            session = result
            print(f"\nSession: {session}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
