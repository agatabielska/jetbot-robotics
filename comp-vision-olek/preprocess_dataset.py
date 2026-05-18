#!/usr/bin/env python3
"""
Preprocess put_jetbot_dataset: apply binary thresholding to every frame,
mark the detected road-center with a circle, and write results to
dataset_preprocessed/ using the same directory structure.

Output CSV format: frame_id, forward, left, cx, cy
  cx, cy — pixel coordinates of the largest white component's centroid,
            or -1 if no component was detected.

Usage:
    python preprocess_dataset.py
    python preprocess_dataset.py --src ../put_jetbot_dataset --dst ../dataset_preprocessed
    python preprocess_dataset.py --threshold 90
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

THRESHOLD_VALUE = 110
MAX_VALUE = 255
CIRCLE_COLOR = (0, 0, 255)   # red in BGR
CIRCLE_RADIUS = 8
CIRCLE_THICKNESS = 2


def apply_binary_threshold(image: np.ndarray, thresh_value: int, max_value: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY)

    h, w = binary.shape
    # Keep only the bottom third (road region)
    binary[: (h * 2) // 3, :] = 0
    # Keep only the middle horizontal third (remove distracting edges)
    binary[:, : w // 3] = 0
    binary[:, (w * 2) // 3 :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def find_largest_component_center(mask: np.ndarray):
    """Return ((cx, cy), area) of the largest white component, or None."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas))
    cx = int(round(centroids[best + 1][0]))
    cy = int(round(centroids[best + 1][1]))
    return (cx, cy), int(areas[best])


def process_session(
    src_img_dir: str,
    src_csv_path: str,
    dst_img_dir: str,
    dst_csv_path: str,
    threshold: int,
) -> int:
    os.makedirs(dst_img_dir, exist_ok=True)

    df = pd.read_csv(src_csv_path, header=None, names=["frame_id", "forward", "left"])
    df["cx"] = -1
    df["cy"] = -1

    for idx, row in df.iterrows():
        frame_id = int(row["frame_id"])
        src_img_path = os.path.join(src_img_dir, f"{frame_id:04d}.jpg")

        if not os.path.exists(src_img_path):
            continue

        image = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        binary = apply_binary_threshold(image, threshold, MAX_VALUE)
        center_info = find_largest_component_center(binary)

        # Convert grayscale binary to BGR so the circle renders in colour
        out_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if center_info is not None:
            (cx, cy), _ = center_info
            df.at[idx, "cx"] = cx
            df.at[idx, "cy"] = cy
            cv2.circle(out_img, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)

        cv2.imwrite(os.path.join(dst_img_dir, f"{frame_id:04d}.jpg"), out_img)

    df.to_csv(dst_csv_path, header=False, index=False)
    return len(df)


def process_split(src_split_dir: str, dst_split_dir: str, threshold: int) -> None:
    os.makedirs(dst_split_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(src_split_dir, "*.csv")))
    if not csv_files:
        print(f"  No CSV files in {src_split_dir}")
        return

    for csv_path in csv_files:
        session = os.path.splitext(os.path.basename(csv_path))[0]
        src_img_dir = os.path.join(src_split_dir, session)

        if not os.path.isdir(src_img_dir):
            print(f"  Skipping {session}: image folder missing")
            continue

        print(f"  {session} ...", end=" ", flush=True)
        n = process_session(
            src_img_dir=src_img_dir,
            src_csv_path=csv_path,
            dst_img_dir=os.path.join(dst_split_dir, session),
            dst_csv_path=os.path.join(dst_split_dir, f"{session}.csv"),
            threshold=threshold,
        )
        print(f"{n} frames")


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=os.path.join(repo_root, "put_jetbot_dataset"),
                   help="Path to the put_jetbot_dataset root (default: ../put_jetbot_dataset)")
    p.add_argument("--dst", default=os.path.join(repo_root, "dataset_preprocessed"),
                   help="Output root (default: ../dataset_preprocessed)")
    p.add_argument("--threshold", type=int, default=THRESHOLD_VALUE,
                   help=f"Binary threshold value (default: {THRESHOLD_VALUE})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dataset = os.path.join(args.src, "dataset")
    dst_dataset = os.path.join(args.dst, "dataset")

    if not os.path.isdir(src_dataset):
        print(f"ERROR: source dataset not found: {src_dataset}")
        sys.exit(1)

    print(f"Source : {src_dataset}")
    print(f"Output : {dst_dataset}")
    print(f"Threshold: {args.threshold}\n")

    for split in ("train", "test"):
        src_split = os.path.join(src_dataset, split)
        dst_split = os.path.join(dst_dataset, split)
        if not os.path.isdir(src_split):
            print(f"Skipping split '{split}': not found")
            continue
        print(f"[{split}]")
        process_split(src_split, dst_split, args.threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
