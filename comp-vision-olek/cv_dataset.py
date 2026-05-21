"""
Dataset loader for CV-based JetBot training.

Reads augmented CSV files from dataset_preprocessed/dataset/{split}/.
CSV format: frame_id, forward, turn, cx, cy

Rows where cx == -1 (no road centre detected) are skipped.
Coordinates are normalised by IMG_SIZE so both inputs lie in [0, 1].
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from cv_models import IMG_SIZE


class CoordDataset(Dataset):
    """
    Each sample: (coords, label)
      coords  – float32 tensor [cx_norm, cy_norm]  in [0, 1]
      label   – float32 tensor [forward, turn]      in [-1, 1]
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: path to dataset_preprocessed/dataset/{split}/
        """
        self.samples = []   # list of (cx_norm, cy_norm, forward, turn)

        csv_files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {root_dir}")

        n_total = 0
        n_skipped = 0
        for csv_path in csv_files:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["frame_id", "forward", "turn", "cx", "cy"],
            )
            n_total += len(df)

            # Drop frames where no road centre was detected
            valid = df[df["cx"] != -1].reset_index(drop=True)
            n_skipped += len(df) - len(valid)

            for _, row in valid.iterrows():
                cx_norm = float(row["cx"]) / IMG_SIZE
                cy_norm = float(row["cy"]) / IMG_SIZE
                self.samples.append((
                    cx_norm,
                    cy_norm,
                    float(row["forward"]),
                    float(row["turn"]),
                ))

        print(
            f"Loaded {len(self.samples)}/{n_total} samples from {len(csv_files)} sessions "
            f"({n_skipped} skipped – no centre detected)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cx_norm, cy_norm, fwd, turn = self.samples[idx]
        coords = torch.tensor([cx_norm, cy_norm], dtype=torch.float32)
        label  = torch.tensor([fwd, turn],        dtype=torch.float32)
        return coords, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    val_split: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset = CoordDataset(root_dir)

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

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
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../dataset_preprocessed/dataset/train"
    train_loader, val_loader = build_dataloaders(root, batch_size=8, num_workers=0)

    coords, labels = next(iter(train_loader))
    print(f"Coords : {coords.shape}  dtype={coords.dtype}")
    print(f"Labels : {labels.shape}  dtype={labels.dtype}")
    print(f"Coord sample  : {coords[:4]}")
    print(f"Label sample  : {labels[:4]}")
