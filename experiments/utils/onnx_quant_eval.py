"""Quantize an ONNX model and compare accuracy (val2) vs the original.

Usage (install deps first):
  pip install onnxruntime onnxruntime-tools pillow numpy torchvision

Examples:
  python utils/onnx_quant_eval.py \
    --onnx bestmodels/run1/best.onnx \
    --quant-path bestmodels/run1/best.quant.onnx \
    --test-dir put_jetbot_dataset/dataset/test \
    --batch-size 64

Notes:
- Uses static QDQ quantization by default, calibrated on the provided test
    dataset. This is more likely to run on the installed ONNX Runtime build than
    dynamic ConvInteger quantization.
- Dynamic quantization is still available as an opt-in fallback.
- The script evaluates only the val2 classification accuracy (3 classes).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    # quantization utilities
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError(
        "Required packages not found. Install with: pip install onnxruntime onnxruntime-tools"
    )

import torchvision.transforms as T


CLASS_TO_VAL = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


class CSVRegressionLoader:
    """Minimal CSV-style dataset loader matching the project's structure.

    Expects directory with subfolders and sibling CSV files named <folder>.csv
    inside the root directory. The CSV first column is image name or index,
    second/third columns are val1 and val2 targets.
    """

    def __init__(self, root_dir: str, transform=None, img_type: str = 'RGB'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_type = img_type
        self.samples: List[Tuple[Path, float, float]] = []

        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = self.root_dir / folder
            if not folder_path.is_dir():
                continue
            csv_path = self.root_dir / f"{folder}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(',')]
                    if len(parts) < 3:
                        continue
                    img_raw = parts[0]
                    try:
                        val1 = float(parts[1])
                        val2 = float(parts[2])
                    except Exception:
                        continue
                    if img_raw.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_name = img_raw
                    else:
                        try:
                            img_name = f"{int(float(img_raw)):04d}.jpg"
                        except Exception:
                            img_name = img_raw
                    img_path = folder_path / img_name
                    if img_path.exists():
                        self.samples.append((img_path, val1, val2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, v1, v2 = self.samples[idx]
        img = Image.open(p).convert(self.img_type)
        if self.transform:
            img = self.transform(img)
        return img, float(v1), float(v2)


class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset: CSVRegressionLoader, input_name: str, batch_size: int = 8, max_samples: int = 0):
        self.dataset = dataset
        self.input_name = input_name
        self.batch_size = batch_size
        self.max_samples = max_samples if max_samples > 0 else len(dataset)
        self._index = 0

    def get_next(self):
        if self._index >= self.max_samples:
            return None

        batch_images = []
        limit = min(self._index + self.batch_size, self.max_samples)
        while self._index < limit:
            img, _, _ = self.dataset[self._index]
            batch_images.append(img.numpy())
            self._index += 1

        if not batch_images:
            return None

        return {self.input_name: batchify(batch_images)}


def make_default_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def batchify(batch_images: List[np.ndarray]) -> np.ndarray:
    return np.stack(batch_images, axis=0).astype(np.float32)


def evaluate_onnx(session: ort.InferenceSession, dataset: CSVRegressionLoader, batch_size: int = 64) -> Tuple[float, int, float, float, float]:
    """Run inference and return (val2_accuracy, num_samples, median_batch_time, total_inference_time, throughput_samples_per_sec)."""
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()

    total = 0
    correct = 0
    timings = []
    total_inference_time = 0.0

    imgs_batch = []
    gt_vals = []

    for i in range(len(dataset)):
        img, v1, v2 = dataset[i]
        imgs_batch.append(img.numpy())
        gt_vals.append(int(round(v2)))
        if len(imgs_batch) >= batch_size or i == len(dataset) - 1:
            batch = batchify(imgs_batch)
            start = time.time()
            ort_outs = session.run(None, {input_name: batch})
            dt = time.time() - start
            timings.append(dt)
            total_inference_time += dt

            # infer val2 logits
            if len(ort_outs) == 1:
                # single-output: assume outputs are concatenated? try to split
                out = ort_outs[0]
                if out.ndim == 2 and out.shape[1] == 4:
                    # odd shape; not expected
                    logits = out[:, 1:4]
                elif out.ndim == 2 and out.shape[1] == 3:
                    logits = out
                else:
                    # fallback: take second half
                    logits = out[:, -3:]
            else:
                # multi-output: pick the largest output as logits or the 2nd
                logits = None
                for o in ort_outs:
                    a = np.asarray(o)
                    if a.ndim == 2 and a.shape[1] == 3:
                        logits = a
                        break
                if logits is None:
                    # fallback to second output
                    logits = np.asarray(ort_outs[1]) if len(ort_outs) > 1 else np.asarray(ort_outs[0])

            preds = np.argmax(logits, axis=1)
            preds_vals = CLASS_TO_VAL[preds]

            for p_val, gt in zip(preds_vals, gt_vals[-len(preds_vals):]):
                if int(round(p_val)) == int(round(gt)):
                    correct += 1
                total += 1

            imgs_batch = []
            gt_vals = []

    acc = correct / total if total > 0 else 0.0
    median_time = float(np.median(timings)) if timings else 0.0
    throughput = float(total) / total_inference_time if total_inference_time > 0 else 0.0
    return acc, total, median_time, total_inference_time, throughput


def quantize_model(
    src: str,
    dst: str,
    dataset: CSVRegressionLoader,
    input_name: str,
    quant_mode: str = 'static',
    weight_type: str = 'QInt8',
    calibration_batch_size: int = 8,
    calibration_samples: int = 128,
) -> None:
    qtype = QuantType.QInt8 if weight_type == 'QInt8' else QuantType.QUInt8
    if quant_mode == 'dynamic':
        quantize_dynamic(src, dst, weight_type=qtype)
        return

    data_reader = ONNXCalibrationDataReader(
        dataset=dataset,
        input_name=input_name,
        batch_size=calibration_batch_size,
        max_samples=calibration_samples,
    )
    quantize_static(
        src,
        dst,
        data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=qtype,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True, help='Path to original ONNX model')
    p.add_argument('--quant-path', required=False, help='Output path for quantized ONNX (optional). If omitted, will be placed next to original with prefix "quantized_"')
    p.add_argument('--test-dir', required=False, default='put_jetbot_dataset/dataset/test', help='Path to test dataset root (CSV structure)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-samples', type=int, default=0, help='Limit number of samples (0 = all)')
    p.add_argument('--quant-mode', choices=['static', 'dynamic'], default='static', help='Quantization mode. static is QDQ calibration-based and usually more compatible.')
    p.add_argument('--calibration-batch-size', type=int, default=8, help='Batch size used for static quantization calibration')
    p.add_argument('--calibration-samples', type=int, default=128, help='Max number of samples used for static quantization calibration')
    p.add_argument('--weight-type', choices=['QInt8','QUInt8'], default='QInt8')
    p.add_argument('--providers', nargs='+', default=None, help='ORT providers order (optional)')
    args = p.parse_args()

    src = Path(args.onnx)
    if args.quant_path:
        dst = Path(args.quant_path)
    else:
        dst = src.parent / f"quantized_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")

    providers = args.providers if args.providers else None

    transform = make_default_transform()
    dataset = CSVRegressionLoader(args.test_dir, transform=transform)
    if args.num_samples > 0:
        dataset.samples = dataset.samples[: args.num_samples]

    print(f'Quantizing ({args.quant_mode}) ...')
    model_probe = ort.InferenceSession(str(src), sess_options=ort.SessionOptions(), providers=providers if providers else None)
    input_name = model_probe.get_inputs()[0].name
    quantize_model(
        str(src),
        str(dst),
        dataset=dataset,
        input_name=input_name,
        quant_mode=args.quant_mode,
        weight_type=args.weight_type,
        calibration_batch_size=args.calibration_batch_size,
        calibration_samples=args.calibration_samples,
    )
    print('Quantization complete.')

    sess_opt = ort.SessionOptions()

    print('Loading original model...')
    sess_orig = ort.InferenceSession(str(src), sess_options=sess_opt, providers=providers if providers else None)
    print('Loading quantized model...')
    sess_q = ort.InferenceSession(str(dst), sess_options=sess_opt, providers=providers if providers else None)

    print(f'Evaluating original model on {len(dataset)} samples...')
    acc_orig, total, tmed_orig, total_time_orig, throughput_orig = evaluate_onnx(sess_orig, dataset, batch_size=args.batch_size)
    print(f'Original val2 accuracy: {acc_orig*100:.2f}%  (samples={total}, median_batch_time={tmed_orig:.4f}s, total_time={total_time_orig:.2f}s, throughput={throughput_orig:.2f} samples/s)')

    print(f'Evaluating quantized model on {len(dataset)} samples...')
    acc_q, total_q, tmed_q, total_time_q, throughput_q = evaluate_onnx(sess_q, dataset, batch_size=args.batch_size)
    print(f'Quantized val2 accuracy: {acc_q*100:.2f}%  (samples={total_q}, median_batch_time={tmed_q:.4f}s, total_time={total_time_q:.2f}s, throughput={throughput_q:.2f} samples/s)')

    delta = (acc_q - acc_orig)
    print(f'Accuracy delta: {delta*100:.2f} percentage points')

    # write results into same folder as quantized model
    results = {
        'original_onnx': str(src),
        'quantized_onnx': str(dst),
        'original_acc': float(acc_orig),
        'quantized_acc': float(acc_q),
        'accuracy_delta': float(delta),
        'orig_median_batch_time': float(tmed_orig),
        'quant_median_batch_time': float(tmed_q),
        'samples': int(total),
    }
    try:
        import json
        out_json = dst.parent / 'quant_eval.json'
        with out_json.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote evaluation summary to {out_json}')
    except Exception:
        pass


if __name__ == '__main__':
    main()
