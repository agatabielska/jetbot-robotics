#!/usr/bin/env python3
"""Simple binary-threshold image tester for the JetBot test folder.

Loads up to five images from the `data-test` directory, applies a binary
threshold, and visualizes each original image side-by-side with its thresholded
version using OpenCV.

This script uses standard OpenCV APIs compatible with `opencv-python>=4.7.0`.
"""

# =============================================================
# THRESHOLD SETTINGS - EDIT THESE VALUES DIRECTLY AT THE TOP
# =============================================================
# BOLD: change THRESHOLD_VALUE to modify the binary threshold.
THRESHOLD_VALUE = 110
MAX_VALUE = 255

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def make_side_by_side(original, thresholded):
    """Concatenate original and thresholded images horizontally."""
    if original.shape[:2] != thresholded.shape[:2]:
        thresholded = cv2.resize(thresholded, (original.shape[1], original.shape[0]))
    return cv2.hconcat([original, cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)])


def find_largest_component_center(mask):
    """Return center and area of the largest connected white component."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    # Skip background label 0 and find the component with the largest area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_area_idx = int(np.argmax(areas))
    label_idx = best_area_idx + 1
    center = centroids[label_idx]
    return (int(round(center[0])), int(round(center[1]))), int(areas[best_area_idx])


def load_image_paths(data_dir, limit=5):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)))
    return image_paths[:limit]


def apply_binary_threshold(image, thresh_value=80, max_value=255):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_value, max_value, cv2.THRESH_BINARY)

    # Keep only the bottom third of the image and set the rest to black.
    height = binary.shape[0]
    bottom_third_start = (height * 2) // 3
    binary[:bottom_third_start, :] = 0

    # Remove white from the left and right thirds of the image.
    width = binary.shape[1]
    left_third_end = width // 3
    right_third_start = (width * 2) // 3
    binary[:, :left_third_end] = 0
    binary[:, right_third_start:] = 0

    # Apply morphological closing followed by opening to clean the mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def visualize_images(image_paths, thresh_value, max_value, save_dir=None):
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    gui_available = True

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue

        thresholded = apply_binary_threshold(image, thresh_value, max_value)
        center_info = find_largest_component_center(thresholded)
        display = make_side_by_side(image, thresholded)

        if center_info is not None:
            (cx, cy), area = center_info
            print(f"  Largest component center: ({cx}, {cy}) area={area}")
            cv2.circle(display, (image.shape[1] + cx, cy), 8, (0, 0, 255), thickness=2)
            cv2.putText(
                display,
                f"center=({cx},{cy}) area={area}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        else:
            print("  No connected component found.")

        title = f"Original | Thresholded  ({idx}/{len(image_paths)})"
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, display)
        except cv2.error as exc:
            print(f"Warning: GUI display unavailable ({exc}).")
            gui_available = False

        if save_dir:
            save_path = os.path.join(save_dir, f"threshold_{idx:02d}.png")
            cv2.imwrite(save_path, display)
            print(f"Saved side-by-side output: {save_path}")

        print(f"Showing: {image_path}")
        if gui_available:
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                print("Exiting early on ESC.")
                break

    if gui_available:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, ".."))
    default_dir = os.path.join(repo_root, "put_jetbot_dataset", "dataset", "train")

    parser = argparse.ArgumentParser(description="Binary threshold tester for up to five train images from put_jetbot_dataset/dataset/train.")
    parser.add_argument("--dir", default=default_dir,
                        help="Path to the train directory containing images")
    parser.add_argument("--count", type=int, default=5,
                        help="Maximum number of images to visualize")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_VALUE,
                        help="Binary threshold value")
    parser.add_argument("--max", type=int, default=MAX_VALUE,
                        help="Maximum value for the thresholded output")
    parser.add_argument("--save", default=None,
                        help="Optional directory to save side-by-side outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.dir)

    if not os.path.isdir(data_dir):
        print(f"ERROR: data-test directory not found: {data_dir}")
        sys.exit(1)

    image_paths = load_image_paths(data_dir, limit=args.count)
    if not image_paths:
        print(f"No images found in: {data_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) in {data_dir}")
    visualize_images(image_paths, args.threshold, args.max, save_dir=args.save)


if __name__ == "__main__":
    main()
