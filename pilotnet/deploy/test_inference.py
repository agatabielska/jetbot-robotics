"""Sanity check that runs on the bot without touching the camera or driver.

Loads model.onnx, runs inference on sample_frame.jpg, prints the
prediction and latency. Hardcoded relative paths -- run from this folder:

    cd pilotnet/deploy
    python3 test_inference.py
"""

import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.onnx")
IMAGE_PATH = os.path.join(HERE, "sample_frame.jpg")
N_RUNS = 100


def main():
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"providers: {sess.get_providers()}")
    print(f"model:  {MODEL_PATH}")
    print(f"image:  {IMAGE_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(IMAGE_PATH)

    chw = preprocess_image(img_bgr)
    batch = chw[None, ...].astype(np.float32)

    out = sess.run(["out"], {"input": batch})[0][0]
    steering, throttle = float(out[0]), float(out[1])
    print(f"prediction:  steering = {steering:+.4f}  throttle = {throttle:+.4f}")

    _ = sess.run(["out"], {"input": batch})
    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run(["out"], {"input": batch})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(
        f"latency over {N_RUNS} runs: "
        f"mean={mean:.2f} ms  p50={p50:.2f} ms  p95={p95:.2f} ms"
    )


if __name__ == "__main__":
    main()
