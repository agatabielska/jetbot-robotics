"""Per-model inference entrypoint with EMA smoothing.

Run from inside the model folder:
    cd .../<model_name>/ && python3 bot_driving.py

Runtime knobs in config.yml (independent of training):
    robot.max_speed      — wheel speed cap
    robot.max_steering   — steering gain
    robot.differential.{left,right}  — per-JetBot calibration
    inference.ema_alpha  — output smoothing
                           1.0 = raw (no smoothing); 0.3 = strong smoothing.
                           Tunable on the lab without retraining.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import yaml

# Local imports (same dir)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preprocess import preprocess
from postprocess import postprocess
from PUTDriver import PUTDriver, gstreamer_pipeline


def _select_providers():
    available = rt.get_available_providers()
    order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


class AI:
    """ONNX model + EMA filter. EMA alpha=1.0 disables smoothing (raw output)."""

    def __init__(self, onnx_path, ema_alpha=1.0):
        providers = _select_providers()
        print("[AI] Loading {}".format(onnx_path))
        print("[AI] Providers (priority): {}".format(providers))
        self.sess = rt.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        # Expected input shape from ONNX (model-specific: e.g. [1, 3, 96, 96] or [1, 3, 224, 224])
        self.expected_shape = tuple(self.sess.get_inputs()[0].shape)
        self.ema_alpha = float(ema_alpha)
        self.last_output = None  # first frame: bypass EMA
        print("[AI] Input shape: {}  ema_alpha={}".format(self.expected_shape, self.ema_alpha))

    def predict(self, img):
        inputs = preprocess(img)
        assert inputs.dtype == np.float32, "preprocess dtype: {}".format(inputs.dtype)
        assert inputs.shape == self.expected_shape, \
            "preprocess shape: {} != expected {}".format(inputs.shape, self.expected_shape)
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = postprocess(detections)
        # EMA smoothing (skip on first frame)
        if self.last_output is not None and self.ema_alpha < 1.0:
            outputs = (self.ema_alpha * outputs +
                       (1.0 - self.ema_alpha) * self.last_output).astype(np.float32)
        self.last_output = outputs.copy()
        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0, "max() = {}".format(outputs.max())
        assert outputs.min() > -1.0, "min() = {}".format(outputs.min())
        return outputs


def main():
    # Load model-local config.yml
    cfg_path = HERE / "config.yml"
    with open(str(cfg_path)) as f:
        config = yaml.safe_load(f)
    ema_alpha = float((config.get("inference") or {}).get("ema_alpha", 1.0))
    print("[main] Model: {}".format(HERE.name))
    print("[main] Config: {}".format(cfg_path))
    print("[main] max_speed={}  max_steering={}  diff=({}, {})  ema_alpha={}".format(
        config["robot"]["max_speed"],
        config["robot"]["max_steering"],
        config["robot"]["differential"]["left"],
        config["robot"]["differential"]["right"],
        ema_alpha,
    ))

    driver = PUTDriver(config=config)
    ai = AI(HERE / "best.onnx", ema_alpha=ema_alpha)

    # Camera frame size = expected ONNX input HxW (so preprocess only color-converts/normalizes)
    H = int(ai.expected_shape[2])
    W = int(ai.expected_shape[3])
    video = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=W, display_height=H),
        cv2.CAP_GSTREAMER,
    )

    ret, frame = video.read()
    if not ret:
        print("[main] No camera")
        return 1
    _ = ai.predict(frame)  # warm-up

    raw_input_fn = input  # py3 builtin
    raw_input_fn("Robot is ready to ride. Press Enter to start...")
    try:
        forward, left = 0.0, 0.0
        while True:
            print("f={:+.4f}  l={:+.4f}".format(forward, left))
            driver.update(float(forward), float(left))
            ret, frame = video.read()
            if not ret:
                print("[main] No camera")
                break
            forward, left = ai.predict(frame)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        video.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
