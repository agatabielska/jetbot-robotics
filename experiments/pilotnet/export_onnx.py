"""Export a trained PilotNet checkpoint to ONNX with a hard parity check
against PyTorch on a real validation image."""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch

from model import PilotNet
from preprocess import INPUT_H, INPUT_W, preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="pilotnet/model.onnx")
    ap.add_argument("--sample_image", required=True)
    ap.add_argument("--max_size_mb", type=float, default=5.0)
    ap.add_argument("--max_abs_diff", type=float, default=1e-4)
    args = ap.parse_args()

    model = PilotNet().cpu().eval()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=13,
        input_names=["input"],
        output_names=["out"],
        dynamic_axes={"input": {0: "N"}, "out": {0: "N"}},
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"onnx file size: {size_mb:.2f} MB")
    assert size_mb < args.max_size_mb, f"ONNX too large: {size_mb:.2f} MB"

    img_bgr = cv2.imread(args.sample_image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.sample_image)
    chw = preprocess_image(img_bgr)
    batch = chw[None, ...].astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(batch)).numpy()

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["out"], {"input": batch})[0]

    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"pytorch out: {pt_out.tolist()}")
    print(f"onnxrt  out: {ort_out.tolist()}")
    print(f"max abs diff: {diff:.2e}")
    assert diff < args.max_abs_diff, f"PT vs ORT mismatch: {diff:.2e}"
    print(f"ok: parity < {args.max_abs_diff:.0e}, size < {args.max_size_mb} MB")


if __name__ == "__main__":
    main()
