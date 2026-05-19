"""JetBot behavioral-cloning dataset.

CSV column 1 is the "left signal" per the course README + PUTDriver source:
  - signed range: [-1.0, +1.0]
  - +1 = LEFT turn, -1 = RIGHT turn, 0 = straight
  - hflip formula: signal = -signal (mirror image swaps left <-> right)

(In code the variable is still called `steering` for historical reasons.
The math doesn't care what we call the axis — mirroring + negating the
label is the correct augmentation in any convention.)
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import preprocess_image


def _apply_color_jitter(
    img_bgr: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.05,
) -> np.ndarray:
    """Color jitter applied on the raw BGR image before crop/resize."""
    img = img_bgr.astype(np.float32)

    b = 1.0 + rng.uniform(-brightness, brightness)
    img = img * b

    c = 1.0 + rng.uniform(-contrast, contrast)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * c + mean

    img = np.clip(img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = 1.0 + rng.uniform(-saturation, saturation)
    hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
    h_shift = rng.uniform(-hue, hue) * 180.0
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180.0
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


class JetBotDataset(Dataset):
    def __init__(
        self,
        samples,
        train: bool = False,
        color_jitter: bool = True,
        hflip: bool = True,
        shift_aug: bool = False,
        seed: int = 0,
    ):
        self.samples = samples
        self.train = train
        self.color_jitter = color_jitter and train
        self.hflip = hflip and train
        self.shift_aug = shift_aug and train
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, steering, throttle = self.samples[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2 could not read {path}")

        if self.color_jitter:
            img_bgr = _apply_color_jitter(img_bgr, self._rng)

        if self.hflip and self._rng.random() < 0.5:
            img_bgr = img_bgr[:, ::-1, :].copy()
            steering = -steering

        chw = preprocess_image(img_bgr)
        return (
            torch.from_numpy(chw),
            torch.tensor([steering, throttle], dtype=torch.float32),
        )


def _debug_dump_batch(samples, out_dir: str, n: int = 4, seed: int = 123) -> None:
    """Write n samples with hflip=False and n with hflip=True, side-by-side.

    Filenames encode the steering label after the flip. Look at the dumps:
      1. flipped image must be a real left-right mirror of the non-flipped one
      2. steering value in the filename must have the opposite sign
    If either fails, hflip is broken and training will not learn left turns.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(samples), size=n, replace=False)

    for i, idx in enumerate(idxs):
        path, steering, throttle = samples[int(idx)]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  WARN: cannot read {path}")
            continue
        flipped = img[:, ::-1, :].copy()
        flipped_steering = -float(steering)

        base = os.path.basename(path).replace(".jpg", "")
        cv2.imwrite(
            os.path.join(out_dir, f"{i:02d}_orig_steer={steering:+.3f}_{base}.png"),
            img,
        )
        cv2.imwrite(
            os.path.join(
                out_dir,
                f"{i:02d}_flip_steer={flipped_steering:+.3f}_{base}.png",
            ),
            flipped,
        )
        print(
            f"  dumped pair {i}: {base}  "
            f"orig steer={steering:+.3f} -> flip steer={flipped_steering:+.3f}"
        )
    print(f"hflip debug dumps -> {out_dir}")
