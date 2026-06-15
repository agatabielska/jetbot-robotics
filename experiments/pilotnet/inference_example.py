"""Deploy template -- runs the trained ONNX model on a single image and
measures latency. NO torch import: deploy stack is onnxruntime + numpy
+ cv2 only.

Output convention: out[0] = steering (signed; +1=right, -1=left, 0=straight),
out[1] = throttle (+ = forward, - = reverse, 0 = stationary).
Adapt to whatever PUTDriver.update(forward, left) expects when wiring on
the bot.
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--n_runs", type=int, default=100)
    args = ap.parse_args()

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(f"providers: {sess.get_providers()}")

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    chw = preprocess_image(img_bgr)
    batch = chw[None, ...].astype(np.float32)

    out = sess.run(["out"], {"input": batch})[0][0]
    steering, throttle = float(out[0]), float(out[1])
    print(f"prediction:  steering = {steering:+.4f}  throttle = {throttle:+.4f}")

    _ = sess.run(["out"], {"input": batch})
    latencies = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        sess.run(["out"], {"input": batch})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(
        f"latency over {args.n_runs} runs: "
        f"mean={mean:.2f} ms  p50={p50:.2f} ms  p95={p95:.2f} ms"
    )


if __name__ == "__main__":
    main()
