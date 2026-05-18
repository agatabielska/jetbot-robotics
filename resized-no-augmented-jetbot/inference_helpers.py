"""
JetBot Inference Helpers — drop these functions into bot_driving.py
───────────────────────────────────────────────────────────────────
The script receives BGR images from the JetBot camera (OpenCV convention).
ONNX runtime always expects channel-first float32 tensors.

Usage in bot_driving.py:
    from inference_helpers import preprocess, postprocess
    ...
    output = session.run(None, {"input": preprocess(bgr_frame)})
    forward, left = postprocess(output)
"""

import cv2
import numpy as np

# ── Constants (must match training) ──────────────────────────────────────────
IMG_SIZE = 96                          # change to 224 if you trained at 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR camera frame into an ONNX-ready input tensor.

    Steps:
      1. Resize to IMG_SIZE × IMG_SIZE
      2. BGR → RGB
      3. Normalise with ImageNet mean/std
      4. HWC → CHW  (channel-first for ONNX)
      5. Add batch dimension → shape (1, 3, H, W)
      6. Cast to float32

    Args:
        bgr_frame: uint8 numpy array of shape (H, W, 3), BGR order.

    Returns:
        float32 numpy array of shape (1, 3, IMG_SIZE, IMG_SIZE).
    """
    # 1. Resize
    img = cv2.resize(bgr_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # 2. BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Normalise to [0,1] then apply ImageNet stats
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD           # shape: (H, W, 3)

    # 4. HWC → CHW
    img = img.transpose(2, 0, 1)      # shape: (3, H, W)

    # 5. Add batch dim
    img = np.expand_dims(img, axis=0)  # shape: (1, 3, H, W)

    return img.astype(np.float32)


def postprocess(onnx_output: list) -> tuple[float, float]:
    """
    Extract and clip the two control signals from the ONNX session output.

    Args:
        onnx_output: list returned by session.run(None, {...})
                     The first element is a (1, 2) float32 array.

    Returns:
        (forward, left) — both clipped to [-1, 1].
    """
    raw = onnx_output[0]               # shape: (1, 2)
    raw = np.array(raw).flatten()      # shape: (2,)
    raw = np.clip(raw, -1.0, 1.0)     # safety clip (tanh should already do this)

    forward = float(raw[0])
    left    = float(raw[1])

    return forward, left


# ── Full inference example ────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test – run this on your PC or JetBot to verify the pipeline:
        python inference_helpers.py path/to/model.onnx path/to/test_image.jpg
    """
    import sys
    import onnxruntime as ort

    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/jetbot_tiny_96.onnx"
    img_path   = sys.argv[2] if len(sys.argv) > 2 else None

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX model: {model_path}")
    print(f"Input  name : {session.get_inputs()[0].name}")
    print(f"Input  shape: {session.get_inputs()[0].shape}")
    print(f"Output name : {session.get_outputs()[0].name}")

    if img_path:
        frame = cv2.imread(img_path)
    else:
        # Random noise frame for smoke test
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("(using random noise frame)")

    import time
    # Warm up
    for _ in range(3):
        session.run(None, {"input": preprocess(frame)})

    # Benchmark
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        out = session.run(None, {"input": preprocess(frame)})
    elapsed = (time.perf_counter() - t0) / N * 1000

    forward, left = postprocess(out)
    print(f"\nInference result:  forward={forward:.4f}  left={left:.4f}")
    print(f"Avg inference time: {elapsed:.2f} ms  ({1000/elapsed:.0f} FPS equivalent)")