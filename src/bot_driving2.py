import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import yaml

from PUTDriver import PUTDriver, gstreamer_pipeline

IMG_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AI:
    def __init__(self, config: dict):
        self.path = config["model"]["path"]
        self.sess = rt.InferenceSession(
            self.path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        out = np.array(detections, dtype=np.float32).flatten()
        if out.size < 2:
            # Keep robot safe if model output is malformed.
            return np.array([0.0, 0.0], dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = np.clip(out, -1.0, 1.0)
        return out[:2].astype(np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)
        return outputs  # [forward, left]


def apply_deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def smooth(previous: float, current: float, alpha: float) -> float:
    return previous * (1.0 - alpha) + current * alpha


def limit_rate(previous: float, target: float, max_step: float) -> float:
    delta = target - previous
    if delta > max_step:
        return previous + max_step
    if delta < -max_step:
        return previous - max_step
    return target


def load_config() -> dict:
    with open("config.yml", "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def main():
    try:
        config = load_config()
    except (OSError, yaml.YAMLError) as exc:
        print(f"Failed to load config.yml: {exc}")
        return

    model_path = Path(config["model"]["path"])
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    deadzone = float(config.get("driving", {}).get("deadzone", 0.03))
    smoothing_alpha = float(config.get("driving", {}).get("smoothing_alpha", 0.35))
    smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
    steering_gain = float(config.get("driving", {}).get("steering_gain", 1.15))
    max_forward = float(config.get("driving", {}).get("max_forward", 1.0))
    max_left = float(config.get("driving", {}).get("max_left", 1.0))
    min_turn_speed_factor = float(
        config.get("driving", {}).get("min_turn_speed_factor", 0.35)
    )
    min_turn_speed_factor = max(0.0, min(1.0, min_turn_speed_factor))
    max_forward_step = float(config.get("driving", {}).get("max_forward_step", 0.06))
    max_left_step = float(config.get("driving", {}).get("max_left_step", 0.08))

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
        cv2.CAP_GSTREAMER,
    )

    if not video_capture.isOpened():
        print("Camera could not be opened")
        return

    ret, image = video_capture.read()
    if not ret:
        print("No camera frame available")
        video_capture.release()
        return

    try:
        _ = ai.predict(image)
    except Exception as exc:
        print(f"Model warmup failed: {exc}")
        video_capture.release()
        return

    input("Robot is ready to ride. Press Enter to start...")

    forward, left = 0.0, 0.0
    frame_idx = 0
    t0 = time.perf_counter()

    try:
        while True:
            ret, image = video_capture.read()
            if not ret:
                print("No camera frame available")
                break

            result = ai.predict(image)
            raw_forward = apply_deadzone(float(result[0]), deadzone)
            raw_left = apply_deadzone(float(result[1]), deadzone)

            # Keep steering responsive, but avoid very sharp steering spikes.
            desired_left = np.clip(raw_left * steering_gain, -max_left, max_left)
            left = smooth(left, float(desired_left), smoothing_alpha)
            left = limit_rate(left, float(desired_left), max_left_step)

            # Auto-slowdown in strong turns to reduce lane overshoot.
            turn_factor = 1.0 - (1.0 - min_turn_speed_factor) * abs(left)
            desired_forward = np.clip(raw_forward * turn_factor, -max_forward, max_forward)
            forward = smooth(forward, float(desired_forward), smoothing_alpha)
            forward = limit_rate(forward, float(desired_forward), max_forward_step)

            # Prefer forward-only driving unless reversing is intentional in config.
            allow_reverse = bool(config.get("driving", {}).get("allow_reverse", False))
            if not allow_reverse:
                forward = max(0.0, forward)

            forward = float(np.clip(forward, -max_forward, max_forward))
            left = float(np.clip(left, -max_left, max_left))
            driver.update(forward, left)

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"Forward: {forward:.4f}\tLeft: {left:.4f}\tFPS: {fps:.1f}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping robot.")
    finally:
        try:
            driver.update(0.0, 0.0)
        except Exception:
            pass
        video_capture.release()


if __name__ == "__main__":
    main()
