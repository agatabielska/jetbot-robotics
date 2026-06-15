"""Per-model inference entrypoint. Self-contained — imports preprocess/postprocess
from the SAME directory (no sys.path tricks).

Run from inside the model folder:
    cd /workspace/models/<this_folder>/ && python3 bot_driving.py
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
    def __init__(self, onnx_path: Path):
        providers = _select_providers()
        print(f"[AI] Loading {onnx_path}")
        print(f"[AI] Providers (priority): {providers}")
        self.sess = rt.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = preprocess(img)
        assert inputs.dtype == np.float32, f"preprocess dtype: {inputs.dtype}"
        assert inputs.shape == (1, 3, 224, 224), f"preprocess shape: {inputs.shape}"
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = postprocess(detections)
        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() < 1.0, f"max() = {outputs.max()}"
        assert outputs.min() > -1.0, f"min() = {outputs.min()}"
        return outputs


def main() -> int:
    # Load model-local config.yml; fall back to mounted /workspace/config.yml if user mapped one.
    local_cfg = HERE / "config.yml"
    mounted_cfg = Path("/workspace/config.yml")
    cfg_path = mounted_cfg if mounted_cfg.exists() and mounted_cfg != local_cfg else local_cfg
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    print(f"[main] Model: {HERE.name}")
    print(f"[main] Config: {cfg_path}")
    print(f"[main] max_speed={config['robot']['max_speed']}  "
          f"diff=({config['robot']['differential']['left']}, {config['robot']['differential']['right']})")

    driver = PUTDriver(config=config)
    ai = AI(HERE / "best.onnx")

    video = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
        cv2.CAP_GSTREAMER,
    )

    ret, frame = video.read()
    if not ret:
        print("[main] No camera"); return 1
    _ = ai.predict(frame)  # warm-up

    input("Robot is ready to ride. Press Enter to start...")
    try:
        forward, left = 0.0, 0.0
        while True:
            print(f"f={forward:+.4f}  l={left:+.4f}")
            driver.update(float(forward), float(left))
            ret, frame = video.read()
            if not ret:
                print("[main] No camera"); break
            forward, left = ai.predict(frame)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        video.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
