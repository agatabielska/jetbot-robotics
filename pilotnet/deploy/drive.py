"""Production driving entry point for the JetBot.

Self-contained: this folder is all you need to deploy. No imports from
sibling folders of the repo.

Camera frame -> preprocess -> ONNX model -> driver.update(forward, left).

Model output convention (CSV/dataset convention, see README.md):
  out[0] = left signal   (+1 = LEFT, -1 = RIGHT)
  out[1] = forward signal (+1 = full forward)

Driver convention (driver.Driver.update):
  left > 0  = turn LEFT
  forward > 0 = drive forward
  → direct pass-through, no sign flip.
"""

import os
import sys
import time
import traceback

import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_image
from driver import Driver, gstreamer_pipeline, load_config


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.yml")


def _resolve_model_path(config: dict) -> str:
    p = config["model"]["path"]
    if not os.path.isabs(p):
        p = os.path.join(HERE, p)
    return p


def _build_session(model_path: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    for accel in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
        if accel in available:
            providers.insert(0, accel)
    return ort.InferenceSession(model_path, providers=providers)


def main():
    config = load_config(CONFIG_PATH)
    model_path = _resolve_model_path(config)
    forward_floor = float(config["drive"]["forward_floor"])

    print(f"loading model: {model_path}")
    sess = _build_session(model_path)
    print(f"providers: {sess.get_providers()}")

    try:
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        driver = Driver(config)
    except Exception:
        print("ERROR: could not initialize camera / driver. Stack trace:")
        traceback.print_exc()
        sys.exit(1)

    if not cap.isOpened():
        print("ERROR: camera did not open. Check gstreamer pipeline.")
        sys.exit(1)

    # Warm-up: pull one frame + one inference so the first loop iter isn't
    # delayed by lazy initialization.
    ok, _frame = cap.read()
    if not ok:
        print("ERROR: first camera read failed.")
        sys.exit(1)
    _ = sess.run(["out"], {"input": np.zeros((1, 3, 66, 200), dtype=np.float32)})

    input("Robot is ready. Press Enter to start driving...")
    print("driving loop start. Ctrl-C to stop.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("WARN: camera read failed, sending stop")
                driver.stop()
                time.sleep(0.05)
                continue

            try:
                chw = preprocess_image(frame_bgr)
                batch = chw[None, ...].astype(np.float32)
                out = sess.run(["out"], {"input": batch})[0][0]
                left_signal = float(np.clip(out[0], -1.0, 1.0))
                forward_signal = float(np.clip(out[1], -1.0, 1.0))
            except Exception:
                print("WARN: inference failed, sending stop")
                traceback.print_exc()
                driver.stop()
                continue

            forward = max(forward_signal, forward_floor)
            driver.update(forward=forward, left=left_signal)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping bot")
    finally:
        driver.stop()
        cap.release()


if __name__ == "__main__":
    main()
