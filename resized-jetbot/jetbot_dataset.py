"""
JetBot Dataset Loader
Loads images + CSV labels from the dataset directory structure.
CSV format: frame_id, forward, left
"""

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 96):
    return A.Compose([
        A.Resize(img_size, img_size),
        # Geometric (mild – don't flip left/right, that would invert steering)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        # Crop bottom rows slightly (hood area) – keeps road visible
        A.RandomCrop(height=int(img_size * 0.9), width=img_size, p=0.3),
        A.Resize(img_size, img_size),   # restore size after crop
        # Lighting / colour – the most impactful augmentations for lane following
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        # Normalise (ImageNet stats work well for pretrained backbones)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 96):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class JetBotDataset(Dataset):
    """
    Reads all session folders under `root_dir`.
    Each session has:
      - a CSV file  <timestamp>.csv  with rows: frame_id, forward, left
      - a folder    <timestamp>/     with images named %04d.jpg
    """

    def __init__(self, root_dir: str, transform=None, future_offset: int = 0):
        """
        Args:
            root_dir:      Path to the dataset/ folder.
            transform:     Albumentations transform pipeline.
            future_offset: If > 0, labels are taken N frames ahead (latency compensation).
        """
        self.transform = transform
        self.future_offset = future_offset
        self.samples = []   # list of (image_path, forward, left)

        csv_files = glob.glob(os.path.join(root_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {root_dir}")

        for csv_path in sorted(csv_files):
            session_name = os.path.splitext(os.path.basename(csv_path))[0]
            img_dir = os.path.join(root_dir, session_name)

            df = pd.read_csv(csv_path, header=None, names=["frame_id", "forward", "left"])
            df["frame_id"] = df["frame_id"].astype(int)
            df = df.sort_values("frame_id").reset_index(drop=True)

            for i, row in df.iterrows():
                img_path = os.path.join(img_dir, f"{int(row.frame_id):04d}.jpg")
                if not os.path.exists(img_path):
                    continue

                # future_offset: use control signal from a later frame as target
                label_idx = min(i + future_offset, len(df) - 1)
                fwd = float(df.loc[label_idx, "forward"])
                left = float(df.loc[label_idx, "left"])

                self.samples.append((img_path, fwd, left))

        print(f"Loaded {len(self.samples)} samples from {len(csv_files)} sessions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fwd, left = self.samples[idx]

        img = np.array(Image.open(img_path).convert("RGB"))   # H x W x 3, RGB

        if self.transform:
            img = self.transform(image=img)["image"]           # C x H x W tensor

        label = torch.tensor([fwd, left], dtype=torch.float32)
        return img, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    img_size: int = 96,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    future_offset: int = 0,
    seed: int = 42,
):
    full_dataset = JetBotDataset(
        root_dir=root_dir,
        transform=None,          # transforms assigned per-split below
        future_offset=future_offset,
    )

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Monkey-patch transforms onto the subset datasets
    train_ds.dataset.transform = get_train_transforms(img_size)
    # val needs its own dataset instance to avoid overwriting train transforms
    val_dataset = JetBotDataset(
        root_dir=root_dir,
        transform=get_val_transforms(img_size),
        future_offset=future_offset,
    )
    val_ds = torch.utils.data.Subset(val_dataset, val_ds.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, sys

    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    train_loader, val_loader = build_dataloaders(root, img_size=96, batch_size=8, num_workers=0)

    imgs, labels = next(iter(train_loader))
    print(f"Batch images : {imgs.shape}   dtype={imgs.dtype}")
    print(f"Batch labels : {labels.shape}  dtype={labels.dtype}")
    print(f"Label sample : {labels[:4]}")