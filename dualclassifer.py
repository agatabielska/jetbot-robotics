#!/usr/bin/env python3
"""Transfer-learning training script using pretrained ResNet18.
Dual-head model:
  - val1: speed classification {0, 0.5, 1} → 3 classes (slow, medium, fast)
  - val2: turn classification {-1, 0, +1}  → 3 classes (left, straight, right)
"""
import argparse
import collections
import os
import random

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Optional OpenCV CLAHE
# ---------------------------------------------------------------------------

try:
    import cv2
    import numpy as np

    class CLAHETransform:
        """Apply CLAHE on the L channel of LAB color space (PIL Image, RGB input).

        Improves local contrast — useful for low-contrast robot camera feeds.
        """
        def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
            self.clip_limit = clip_limit
            self.tile_grid_size = tile_grid_size

        def __call__(self, img: Image.Image) -> Image.Image:
            arr = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            if arr.ndim == 2:
                return Image.fromarray(clahe.apply(arr))
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            merged = cv2.merge((clahe.apply(l), a, b))
            return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))

except Exception:
    CLAHETransform = None


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

class BottomHalfResize:
    """Crop the bottom `fraction` of the image (full width), then resize to `size`×`size`.

    Keeps road/track information and discards uninformative sky/ceiling.
    """
    def __init__(self, fraction: float = 0.5, size: int = 224):
        self.fraction = fraction
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        top = int(h * (1.0 - self.fraction))
        return img.crop((0, top, w, h)).resize((self.size, self.size), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

# val1 (speed):  raw {0.0, 0.5, 1.0} → class index {0, 1, 2}
# val2 (turn):   raw {-1,  0,  +1}   → class index {0, 1, 2}

SPEED_VALUES = torch.tensor([0.0, 0.5, 1.0])   # index → motor value
TURN_VALUES  = torch.tensor([-1.0, 0.0, 1.0])  # index → motor value


def bucket_speed(val: float) -> float:
    """Snap a raw speed value to the nearest class centre {0.0, 0.5, 1.0}."""
    if val < 0.33:
        return 0.0
    elif val < 0.67:
        return 0.5
    return 1.0


def speed_to_cls(tensor_col: torch.Tensor) -> torch.Tensor:
    """Convert speed labels (0.0/0.5/1.0) → class indices (0/1/2)."""
    return (tensor_col * 2).round().long().clamp(0, 2)


def turn_to_cls(tensor_col: torch.Tensor) -> torch.Tensor:
    """Convert turn labels (-1/0/+1) → class indices (0/1/2)."""
    return (tensor_col + 1).round().long().clamp(0, 2)


def cls_to_speed(cls: torch.Tensor, device: torch.device) -> torch.Tensor:
    return SPEED_VALUES.to(device)[cls]


def cls_to_turn(cls: torch.Tensor, device: torch.device) -> torch.Tensor:
    return TURN_VALUES.to(device)[cls]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualHeadResNet(nn.Module):
    """ResNet18 backbone with two classification heads:
      - head_speed: 3-class speed  {slow, medium, fast}
      - head_turn:  3-class turn   {left, straight, right}
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features          # 512
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Shared layer — forces both heads to agree on a common representation.
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3),   # raw logits → 3 classes
            )

        self.head_speed = _head()
        self.head_turn  = _head()

    def forward(self, x: torch.Tensor):
        feat  = self.backbone(x).flatten(1)       # (B, 512)
        shared = self.shared(feat)                 # (B, 256)
        return self.head_speed(shared), self.head_turn(shared)   # (B,3), (B,3)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DualLoss(nn.Module):
    """CrossEntropy for each head, weighted by inverse class frequency."""

    def __init__(self, speed_counts: dict, turn_counts: dict):
        """
        Args:
            speed_counts: {0: n_slow, 1: n_medium, 2: n_fast}
            turn_counts:  {0: n_left, 1: n_straight, 2: n_right}
        """
        super().__init__()
        self.ce_speed = nn.CrossEntropyLoss(weight=self._inv_freq_weights(speed_counts))
        self.ce_turn  = nn.CrossEntropyLoss(weight=self._inv_freq_weights(turn_counts))

    @staticmethod
    def _inv_freq_weights(counts: dict) -> torch.Tensor:
        if not counts:
            return torch.ones(3)
        vals  = [counts.get(k, 1) for k in range(3)]   # keys are always 0,1,2
        total = sum(vals)
        return torch.tensor([total / (3.0 * v) for v in vals], dtype=torch.float32)

    def forward(
        self,
        speed_logits: torch.Tensor,   # (B, 3)
        turn_logits:  torch.Tensor,   # (B, 3)
        targets:      torch.Tensor,   # (B, 2)  col0=speed, col1=turn
    ) -> torch.Tensor:
        speed_cls = speed_to_cls(targets[:, 0])
        turn_cls  = turn_to_cls(targets[:, 1])
        return self.ce_speed(speed_logits, speed_cls) + self.ce_turn(turn_logits, turn_cls)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RandomHorizontalFlipWithLabel:
    """Horizontal flip that also negates turn direction (steering symmetry).

    Speed is NOT negated — flipping the image doesn't change required speed.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, targets: torch.Tensor):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            targets = targets * torch.tensor([1.0, -1.0])   # negate turn only
        return image, targets


class CSVRegressionDataset(Dataset):
    """Loads (image, [speed, turn]) pairs from per-folder CSVs.

    CSV format: col0=image_name, col1=speed (0/0.5/1), col2=turn (-1/0/+1).
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        img_type: str = 'RGB',
        flip_augment: bool = False,
    ):
        self.root_dir     = root_dir
        self.transform    = transform
        self.img_type     = img_type
        self.flip_augment = flip_augment
        self.flipper      = RandomHorizontalFlipWithLabel(p=0.5)
        self.data_samples: list[tuple[str, list[float]]] = []

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            csv_path = os.path.join(root_dir, f"{folder_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: missing CSV for '{folder_name}' at {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_raw = str(row.iloc[0])
                speed   = bucket_speed(float(row.iloc[1]))   # snap to 0/0.5/1
                turn    = float(row.iloc[2])

                if img_raw.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = img_raw
                else:
                    try:
                        img_name = f'{int(float(img_raw)):04d}.jpg'
                    except Exception:
                        img_name = img_raw

                img_path = os.path.join(folder_path, img_name)
                if os.path.exists(img_path):
                    self.data_samples.append((img_path, [speed, turn]))

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        img_path, targets = self.data_samples[idx]
        image   = Image.open(img_path).convert(self.img_type)
        targets = torch.tensor(targets, dtype=torch.float32)

        if self.flip_augment:
            image, targets = self.flipper(image, targets)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image.resize((224, 224)))

        return image, targets

    def class_counts(self) -> tuple[dict, dict]:
        """Return (speed_counts, turn_counts) as {class_index: count} dicts."""
        speed_counts: dict[int, int] = collections.Counter()
        turn_counts:  dict[int, int] = collections.Counter()
        for _, targets in self.data_samples:
            speed_counts[int(round(targets[0] * 2))] += 1          # 0.0→0, 0.5→1, 1.0→2
            turn_counts[int(round(targets[1])) + 1]  += 1          # -1→0, 0→1, +1→2
        return dict(speed_counts), dict(turn_counts)


# ---------------------------------------------------------------------------
# Balanced oversampling (by turn class, as before)
# ---------------------------------------------------------------------------

class BalancedOversampledDataset(Dataset):
    """Resample training set to `target_per_class` examples per turn class."""

    def __init__(
        self,
        base_dataset: CSVRegressionDataset,
        target_per_class: int = 4000,
        seed: int = 42,
    ):
        self.transform    = base_dataset.transform
        self.img_type     = base_dataset.img_type
        self.flip_augment = base_dataset.flip_augment
        self.flipper      = base_dataset.flipper

        buckets: dict[int, list] = {-1: [], 0: [], 1: []}
        for img_path, targets in base_dataset.data_samples:
            cls = int(round(targets[1]))
            buckets[max(-1, min(1, cls))].append((img_path, targets))

        random.seed(seed)
        balanced = []
        for cls in (-1, 0, 1):
            samples = buckets[cls]
            n = len(samples)
            if n == 0:
                continue
            if n >= target_per_class:
                chosen = random.sample(samples, target_per_class)
            else:
                chosen = list(samples) + [random.choice(samples) for _ in range(target_per_class - n)]
            balanced.extend(chosen)

        random.shuffle(balanced)
        self.data_samples = balanced

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        img_path, targets = self.data_samples[idx]
        image   = Image.open(img_path).convert(self.img_type)
        targets = torch.tensor(targets, dtype=torch.float32)

        if self.flip_augment:
            image, targets = self.flipper(image, targets)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image.resize((224, 224)))

        return image, targets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model:     DualHeadResNet,
    loader:    DataLoader,
    criterion: DualLoss,
    device:    torch.device,
    epoch:     int = None,
) -> tuple[float, float, float]:
    """Evaluate model.

    Returns:
        (avg_loss, speed_accuracy, turn_accuracy)
    """
    model.eval()
    running_loss   = 0.0
    num_samples    = 0
    speed_correct  = 0
    turn_correct   = 0

    label = f'Epoch {epoch} eval' if epoch is not None else 'Evaluating'

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(loader, desc=label, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            speed_logits, turn_logits = model(imgs)
            loss = criterion(speed_logits, turn_logits, labels)

            b = imgs.size(0)
            running_loss  += loss.item() * b
            num_samples   += b

            speed_pred = speed_logits.argmax(dim=1)
            turn_pred  = turn_logits.argmax(dim=1)

            speed_gt = speed_to_cls(labels[:, 0])
            turn_gt  = turn_to_cls(labels[:, 1])

            speed_correct += (speed_pred == speed_gt).sum().item()
            turn_correct  += (turn_pred  == turn_gt).sum().item()

    avg_loss   = running_loss / num_samples if num_samples > 0 else float('nan')
    speed_acc  = speed_correct / num_samples if num_samples > 0 else 0.0
    turn_acc   = turn_correct  / num_samples if num_samples > 0 else 0.0

    return avg_loss, speed_acc, turn_acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if CLAHETransform is not None:
        print(f'CLAHE: clip={args.clahe_clip}, grid={args.clahe_grid}×{args.clahe_grid}')
        clahe = CLAHETransform(clip_limit=args.clahe_clip, tile_grid_size=(args.clahe_grid, args.clahe_grid))
    else:
        clahe = T.Lambda(lambda x: x)

    train_transform = T.Compose([
        T.Resize(256),
        BottomHalfResize(fraction=args.crop_fraction, size=224),
        T.RandomRotation(degrees=5),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=1, saturation=0.2, hue=0.2)], p=0.5),
        clahe,
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5))], p=0.5),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize(256),
        BottomHalfResize(fraction=args.crop_fraction, size=224),
        clahe,
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- datasets ---
    base_train = CSVRegressionDataset(root_dir=args.data_dir, transform=train_transform, flip_augment=True)
    speed_counts, turn_counts = base_train.class_counts()
    print(f'Speed class counts (0=slow,1=med,2=fast): {speed_counts}')
    print(f'Turn  class counts (0=left,1=str,2=rght): {turn_counts}')

    if args.target_per_class > 0:
        print(f'Balancing by turn class → {args.target_per_class} samples/class')
        train_dataset = BalancedOversampledDataset(base_train, target_per_class=args.target_per_class)
    else:
        train_dataset = base_train

    print(f'Batch size: {args.batch_size}  |  Train samples: {len(train_dataset)}')

    train_loader      = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_dir = os.path.join(os.path.dirname(args.data_dir.rstrip('/\\')), 'test')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found at '{test_dir}'.")

    test_dataset = CSVRegressionDataset(root_dir=test_dir, transform=test_transform, flip_augment=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'Test samples: {len(test_dataset)}')

    # --- model, loss, optimizer ---
    model     = DualHeadResNet(pretrained=True, dropout=args.dropout).to(device)
    criterion = DualLoss(speed_counts=speed_counts, turn_counts=turn_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor,
                                  patience=args.scheduler_patience, verbose=True)

    if args.freeze_epochs > 0:
        print(f'Freezing backbone for first {args.freeze_epochs} epoch(s).')
        for param in model.backbone.parameters():
            param.requires_grad = False

    best_test_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print('Unfreezing backbone.')
            for param in model.backbone.parameters():
                param.requires_grad = True

        # --- train loop ---
        model.train()
        running_loss = 0.0
        loop = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            speed_logits, turn_logits = model(imgs)
            loss = criterion(speed_logits, turn_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=f'{running_loss / ((loop.n + 1) * args.batch_size):.4f}')

        train_loss = running_loss / len(train_dataset)

        # --- evaluate ---
        test_loss, test_speed_acc, test_turn_acc = run_evaluation(model, test_loader, criterion, device, epoch)

        do_train_eval = (epoch == 1 or epoch % 5 == 0)
        if do_train_eval:
            _, tr_speed_acc, tr_turn_acc = run_evaluation(model, train_eval_loader, criterion, device, epoch)
            print(
                f'  speed acc — train: {tr_speed_acc*100:.1f}%  test: {test_speed_acc*100:.1f}%  |  '
                f'turn  acc — train: {tr_turn_acc*100:.1f}%  test: {test_turn_acc*100:.1f}%'
            )
        else:
            print(
                f'  speed acc — test: {test_speed_acc*100:.1f}%  |  '
                f'turn  acc — test: {test_turn_acc*100:.1f}%'
            )

        scheduler.step(test_loss)

        is_best = test_loss < best_test_loss
        print(
            f'Epoch {epoch:>3}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}'
            + (' ← best' if is_best else '')
        )

        if is_best:
            best_test_loss = test_loss
            if args.output:
                torch.save(model.state_dict(), args.output)
                print(f'  Saved best model → {args.output}')

        if args.checkpoint:
            os.makedirs(args.checkpoint, exist_ok=True)
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss':      train_loss,
                'test_loss':       test_loss,
                'test_speed_acc':  test_speed_acc,
                'test_turn_acc':   test_turn_acc,
            }, os.path.join(args.checkpoint, f'resnet18_epoch{epoch:03d}.pt'))

    print(f'\nTraining complete. Best test loss: {best_test_loss:.4f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',           type=str,   default='put_jetbot_dataset/dataset/train')
    parser.add_argument('--epochs',             type=int,   default=50)
    parser.add_argument('--batch-size',         type=int,   default=128)
    parser.add_argument('--lr',                 type=float, default=1e-3)
    parser.add_argument('--dropout',            type=float, default=0.3)
    parser.add_argument('--weight-decay',       type=float, default=1e-4)
    parser.add_argument('--scheduler-factor',   type=float, default=0.3)
    parser.add_argument('--scheduler-patience', type=int,   default=2)
    parser.add_argument('--clahe-clip',         type=float, default=5.0)
    parser.add_argument('--clahe-grid',         type=int,   default=15)
    parser.add_argument('--crop-fraction',      type=float, default=0.5,
                        help='Bottom fraction of image to keep (0.4–0.6 recommended)')
    parser.add_argument('--target-per-class',   type=int,   default=4000,
                        help='Samples per turn class after balancing (0 = no balancing)')
    parser.add_argument('--num-workers',        type=int,   default=4)
    parser.add_argument('--freeze-epochs',      type=int,   default=0)
    parser.add_argument('--output',             type=str,   help='Path to save best model weights')
    parser.add_argument('--checkpoint',         type=str,   help='Directory for per-epoch checkpoints')
    args = parser.parse_args()

    train(args)