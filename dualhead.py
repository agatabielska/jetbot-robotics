#!/usr/bin/env python3
"""Transfer-learning training script using pretrained ResNet18.
Dual-head model: val1 as regression (tanh), val2 as 3-class classification {-1, 0, +1}.
"""
import argparse
import os
import collections

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import pandas as pd
import time

from utils.export_model import export_run_onnx
from utils.confusion import print_confusion_matrix


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualHeadResNet(nn.Module):
    """ResNet18 backbone with two separate heads:
      - head_val1: regression → tanh output in (-1, 1)
      - head_val2: 3-class classifier → {-1, 0, +1}
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc

        self.head_val1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.head_val2 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),  # logits for classes: -1, 0, +1
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).flatten(1)      # (B, 512)
        val1 = self.head_val1(feat).squeeze(1)  # (B,)
        val2_logits = self.head_val2(feat)       # (B, 3)
        return val1, val2_logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DualLoss(nn.Module):
    """Combined loss:
      - val1: MSE regression
      - val2: CrossEntropy classification with inverse-frequency class weights
    """

    def __init__(self, val2_counts):
        """
        Args:
            val2_counts: dict mapping val2 value → count, e.g. {-1: 1468, 0: 3749, 1: 1405}.
                         If None, uses uniform weights.
        """
        super().__init__()
        self.mse = nn.MSELoss()

        if val2_counts is not None:
            n_neg = val2_counts.get(-1, 1)
            n_zer = val2_counts.get(0,  1)
            n_pos = val2_counts.get(1,  1)
            total = n_neg + n_zer + n_pos
            weights = torch.tensor([
                total / (3.0 * n_neg),
                total / (3.0 * n_zer),
                total / (3.0 * n_pos),
            ])
        else:
            weights = torch.ones(3)

        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(
        self,
        val1_pred: torch.Tensor,    # (B,)
        val2_logits: torch.Tensor,  # (B, 3)
        targets: torch.Tensor,      # (B, 2)
    ) -> torch.Tensor:
        loss_val1 = self.mse(val1_pred, targets[:, 0])

        # convert continuous val2 label → class index: -1→0, 0→1, +1→2
        val2_cls = (targets[:, 1] + 1).round().long().clamp(0, 2)
        loss_val2 = self.ce(val2_logits, val2_cls)

        return loss_val1 + loss_val2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RandomHorizontalFlipWithLabel:
    """Horizontal flip that also negates both val1 and val2 (steering symmetry)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, targets: torch.Tensor):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            targets = targets * torch.tensor([-1.0, -1.0])
        return image, targets


class CSVRegressionDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, img_type: str = 'RGB', flip_augment: bool = False):
        """
        Args:
            root_dir:      Path to train (or test) folder.
            transform:     Torchvision transform applied to the image only.
            img_type:      PIL image mode ('RGB' or 'L').
            flip_augment:  Apply random horizontal flip with label negation (train only).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_type = img_type
        self.flip_augment = flip_augment
        self.flipper = RandomHorizontalFlipWithLabel(p=0.5)
        self.data_samples: list[tuple[str, list[float]]] = []

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            csv_path = os.path.join(root_dir, f"{folder_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: missing CSV for folder '{folder_name}' at {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_raw = str(row.iloc[0])
                val1 = float(row.iloc[1])
                val2 = float(row.iloc[2])

                if img_raw.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = img_raw
                else:
                    try:
                        img_name = f'{int(float(img_raw)):04d}.jpg'
                    except Exception:
                        img_name = img_raw

                img_path = os.path.join(folder_path, img_name)
                if os.path.exists(img_path):
                    self.data_samples.append((img_path, [val1, val2]))

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        img_path, targets = self.data_samples[idx]
        image = Image.open(img_path).convert(self.img_type)
        targets = torch.tensor(targets, dtype=torch.float32)

        # flip before tensor pipeline (needs PIL image)
        if self.flip_augment:
            image, targets = self.flipper(image, targets)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image.resize((224, 224)))

        return image, targets

    def val2_counts(self) -> dict:
        """Return counts of rounded val2 values for loss weighting."""
        counts: dict[int, int] = collections.Counter()
        for _, (_, targets) in enumerate(self.data_samples):
            val2 = targets[1]
            key = int(round(val2))  # -1, 0, or 1
            counts[key] += 1
        return dict(counts)


class BalancedOversampledDataset(Dataset):
    """Create a balanced dataset with exactly `target_per_class` samples per val2 class (-1,0,1).

    It takes an existing `CSVRegressionDataset` (or any dataset exposing `data_samples` as a
    list of (img_path, [val1,val2])) and builds a new list of samples by up/down-sampling
    each class to match `target_per_class`.
    """
    def __init__(self, base_dataset: CSVRegressionDataset, target_per_class: int = 4000, seed: int = 42):
        self.transform = base_dataset.transform
        self.img_type = base_dataset.img_type
        self.flip_augment = base_dataset.flip_augment
        self.flipper = base_dataset.flipper  # reuse the same flipper instance

        # collect samples per class
        buckets = { -1: [], 0: [], 1: [] }
        for img_path, targets in base_dataset.data_samples:
            cls = int(round(targets[1]))
            cls = max(-1, min(1, cls))
            buckets[cls].append((img_path, targets))

        import random
        random.seed(seed)

        balanced = []
        for cls in (-1, 0, 1):
            samples = buckets[cls]
            n = len(samples)
            if n == 0:
                continue
            if n >= target_per_class:
                # downsample without replacement
                chosen = random.sample(samples, target_per_class)
            else:
                # upsample with replacement
                chosen = list(samples)
                needed = target_per_class - n
                for _ in range(needed):
                    chosen.append(random.choice(samples))
            balanced.extend(chosen)

        # shuffle final list
        random.shuffle(balanced)

        self.data_samples = balanced

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx: int):
        img_path, targets = self.data_samples[idx]
        image = Image.open(img_path).convert(self.img_type)
        targets = torch.tensor(targets, dtype=torch.float32)

        if self.flip_augment:
            image, targets = self.flipper(image, targets)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image.resize((224, 224)))

        return image, targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_TO_VAL = torch.tensor([-1.0, 0.0, 1.0])


def logits_to_val2(logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert val2 class logits → scalar values {-1, 0, +1}."""
    cls = logits.argmax(dim=1)
    return CLASS_TO_VAL.to(device)[cls]


def smooth_labels(targets: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """Pull val1 labels slightly away from ±1 to avoid tanh saturation."""
    result = targets.clone()
    result[:, 0] = result[:, 0] * (1.0 - eps)
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model: DualHeadResNet,
    loader: DataLoader,
    criterion: DualLoss,
    device: torch.device,
    epoch: int = None,
    print_samples: bool = False,
) -> tuple[float, float]:
    """Evaluate model and return (average loss, val2_accuracy).

    Args:
        print_samples: whether to print a few sample predictions from the last batch.
    Returns:
        (avg_loss, val2_accuracy)
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    label = f'Epoch {epoch} eval' if epoch is not None else 'Evaluating'
    t_start = time.time()

    total_correct = 0
    total_seen = 0
    last_batch_out = None
    last_batch_gt = None
    all_pred_cls: list[int] = []
    all_gt_cls: list[int] = []

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(loader, desc=label, leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = smooth_labels(labels, eps=0.05)  # consistent with training loss scale

            val1_pred, val2_logits = model(imgs)
            loss = criterion(val1_pred, val2_logits, labels)

            running_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)

            # compute val2 accuracy for this batch
            val2_pred_vals = logits_to_val2(val2_logits, device)  # -1/0/1
            pred_cls = (val2_pred_vals.round().long() + 1).clamp(0, 2)
            gt_cls = ((labels[:, 1] + 1).round().long()).clamp(0, 2)
            total_correct += (pred_cls == gt_cls).sum().item()
            total_seen += imgs.size(0)

            last_batch_out = val2_pred_vals
            last_batch_gt = labels

            all_pred_cls.extend(pred_cls.cpu().tolist())
            all_gt_cls.extend(gt_cls.cpu().tolist())

    avg_loss = running_loss / num_samples if num_samples > 0 else float('nan')
    acc = float(total_correct) / float(total_seen) if total_seen > 0 else 0.0

    t_end = time.time()
    print(f"Evaluation time: {t_end - t_start:.2f}s over {num_samples} samples")

    if print_samples and last_batch_out is not None:
        print(f"\n  {'Pred[0]':>10} {'Pred[1]':>10} {'GT[0]':>10} {'GT[1]':>10}")
        for i in range(min(5, last_batch_out.size(0))):
            print(
                f"  {0.0:>10.4f}"  # placeholder for val1, not shown here
                f" {last_batch_out[i].item():>10.4f}"
                f" {last_batch_gt[i, 0].item():>10.4f}"
                f" {last_batch_gt[i, 1].item():>10.4f}"
            )

        # Print confusion matrix for val2 (map class idx 0->-1,1->0,2->1)
        if len(all_pred_cls) > 0:
            label_map = [-1, 0, 1]
            y_pred_vals = [label_map[c] for c in all_pred_cls]
            y_true_vals = [label_map[c] for c in all_gt_cls]
            print_confusion_matrix(y_true_vals, y_pred_vals, labels=[-1, 0, 1], normalize=False)

    return avg_loss, acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # Performance: enable cudnn autotuner when using fixed-size inputs on CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    pin_memory = (device.type == 'cuda')

    train_transform = T.Compose([
        T.Resize(256),
        BottomHalfResize(fraction=args.crop_fraction, size=224),
        # Geometric augmentation: prevents spatial memorization
        T.RandomRotation(degrees=15),
        T.RandomPerspective(distortion_scale=0.1, p=0.15),
        # Colour augmentation
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.15),
        ], p=0.5),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        ], p=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    test_transform = T.Compose([
        T.Resize(256),
        BottomHalfResize(fraction=args.crop_fraction, size=224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- datasets ---
    base_train_dataset = CSVRegressionDataset(
        root_dir=args.data_dir,
        transform=train_transform,
        flip_augment=True,
    )

    # compute val2 class counts from the original data (before balancing)
    counts = base_train_dataset.val2_counts()
    print(f'val2 class counts (original): {counts}')

    # Optionally balance the training set to have exactly target_per_class samples per val2 class
    if getattr(args, 'target_per_class', 0) and args.target_per_class > 0:
        print(f'Balancing training set to {args.target_per_class} samples per val2 class')
        train_dataset = BalancedOversampledDataset(base_train_dataset, target_per_class=args.target_per_class)
    else:
        train_dataset = base_train_dataset

    print(f'Batch size: {args.batch_size}  |  Train samples: {len(train_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    # evaluation loader for train (no shuffling) to compute full-dataset metrics
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    test_dir = os.path.join(os.path.dirname(args.data_dir.rstrip('/\\')), 'test')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found at '{test_dir}'. "
            f"Expected it adjacent to '{args.data_dir}'."
        )
    test_dataset = CSVRegressionDataset(
        root_dir=test_dir,
        transform=test_transform,
        flip_augment=False,
    )
    print(f'Test samples: {len(test_dataset)}')
    # Use larger batch size for evaluation (no backprop, so memory is not constrained)
    eval_batch_size = min(256, args.batch_size * 4)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    # --- model ---
    model = DualHeadResNet(pretrained=True, dropout=args.dropout).to(device)
    criterion = DualLoss(val2_counts=counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True)

    # optional: freeze backbone for first few epochs
    if args.freeze_epochs > 0:
        print(f'Freezing backbone for first {args.freeze_epochs} epoch(s).')
        for param in model.backbone.parameters():
            param.requires_grad = False

    best_test_loss = float('inf')
    best_epoch = None
    best_test_acc = float('-inf')

    for epoch in range(1, args.epochs + 1):

        # unfreeze backbone after freeze_epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print('Unfreezing backbone.')
            for param in model.backbone.parameters():
                param.requires_grad = True

        # --- train ---
        model.train()
        running_loss = 0.0
        loop = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')

        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = smooth_labels(labels, eps=0.05)  # avoid tanh saturation on val1

            optimizer.zero_grad()
            val1_pred, val2_logits = model(imgs)
            loss = criterion(val1_pred, val2_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=running_loss / ((loop.n + 1) * args.batch_size))

        train_loss_loop = running_loss / len(train_dataset)

        # --- evaluate on test and train sets ---
        # Always evaluate train with no-grad / eval mode for a fair comparison with test loss
        if not args.skip_train_eval:
            train_loss_eval, train_acc = run_evaluation(model, train_eval_loader, criterion, device, epoch, print_samples=False)
        else:
            train_loss_eval, train_acc = float('nan'), float('-inf')
        
        test_loss, test_acc = run_evaluation(model, test_loader, criterion, device, epoch, print_samples=(not args.no_confusion))
        if not args.skip_train_eval:
            print(f"  val2 accuracy - train: {train_acc * 100:.2f}%, test: {test_acc * 100:.2f}%")
        else:
            print(f"  val2 accuracy - test: {test_acc * 100:.2f}%")
        # step scheduler based on validation loss
        try:
            scheduler.step(test_loss)
        except Exception:
            pass

        is_best = test_acc > best_test_acc

        # Use train_loss_eval (eval-mode, consistent with test_loss) for the gap metric
        if not args.skip_train_eval:
            gap = train_loss_eval - test_loss
            print(
                f'Epoch {epoch:>3}  train_loss(loop)={train_loss_loop:.4f}  '
                f'train_loss(eval)={train_loss_eval:.4f}  test_loss={test_loss:.4f}  '
                f'gap={gap:+.4f}'
                + (' ← best' if is_best else '')
            )
        else:
            print(
                f'Epoch {epoch:>3}  train_loss(loop)={train_loss_loop:.4f}  '
                f'test_loss={test_loss:.4f}'
                + (' ← best' if is_best else '')
            )
        if is_best:
            best_test_loss = test_loss
            best_epoch = epoch
            best_test_acc = test_acc
            if args.output:
                torch.save(model.state_dict(), args.output)
                print(f'  Saved best model → {args.output}')

            if args.export_onnx:
                export_paths = export_run_onnx(
                    model=model,
                    sample_input=torch.randn(1, 3, 224, 224, device=device),
                    run_name=args.run_name,
                    role='best',
                    test_transform=test_transform,
                    metadata={
                        'best_epoch': epoch,
                        'best_test_loss': float(test_loss),
                        'best_test_accuracy': float(test_acc),
                        'train_loss_current_epoch': float(train_loss_eval),
                    },
                    output_root=args.onnx_output_root,
                    opset_version=args.onnx_opset,
                )
                print(f"  Exported best ONNX → {export_paths['onnx_path']}")

        if args.checkpoint:
            os.makedirs(args.checkpoint, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint, f'resnet18_epoch{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss_eval if not args.skip_train_eval else train_loss_loop,
                'test_loss': test_loss,
            }, ckpt_path)

    print(f'\nTraining complete. Best test loss: {best_test_loss:.4f}')
    if best_epoch is not None and best_test_acc is not None:
        print(f'Best epoch: {best_epoch} | Best test accuracy: {best_test_acc * 100:.2f}%')
    # export last-epoch model as well
    if args.export_onnx:
        export_paths = export_run_onnx(
            model=model,
            sample_input=torch.randn(1, 3, 224, 224, device=device),
            run_name=args.run_name,
            role='last',
            test_transform=test_transform,
            metadata={
                'final_epoch': args.epochs,
                'final_test_loss': float(best_test_loss),
            },
            output_root=args.onnx_output_root,
            opset_version=args.onnx_opset,
        )
        print(f"Exported last-epoch ONNX → {export_paths['onnx_path']}")


# ---------------------------------------------------------------------------
# Entry point
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      type=str,   default='put_jetbot_dataset/dataset/train',
                        help='Path to train folder; test folder inferred as adjacent "test" dir')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--dropout',       type=float, default=0.5)
    parser.add_argument('--weight-decay',  type=float, default=1e-2, help='Weight decay for AdamW')
    parser.add_argument('--scheduler-factor', type=float, default=0.3, help='ReduceLROnPlateau factor')
    parser.add_argument('--scheduler-patience', type=int, default=2, help='ReduceLROnPlateau patience')
    parser.add_argument('--target-per-class', type=int, default=4000, help='If >0, up/down-sample train set to this many samples per val2 class')
    parser.add_argument('--num-workers',   type=int,   default=4)
    parser.add_argument('--freeze-epochs', type=int,   default=5,
                        help='Freeze backbone for this many epochs before unfreezing')
    parser.add_argument('--output',        type=str,   help='Path to save best model state dict')
    parser.add_argument('--checkpoint',    type=str,   help='Directory to save per-epoch checkpoints')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export ONNX for each new best epoch (with transform + metadata sidecar files)')
    parser.add_argument('--onnx-output-root', type=str, default='bestmodels',
                        help='Root directory for ONNX exports and sidecar files')
    parser.add_argument('--onnx-opset', type=int, default=11,
                        help='ONNX opset version to use for export')
    parser.add_argument('--run-name', type=str, default='run1',
                        help='Run folder name under onnx output root (e.g. run1)')
    parser.add_argument('--skip-train-eval', action='store_true',
                        help='Skip train set evaluation to speed up training (only evaluate test set)')
    parser.add_argument('--no-confusion', action='store_true',
                        help='Skip confusion matrix printing (speeds up eval output)')
    parser.add_argument('--crop-fraction',      type=float, default=0.7,
                        help='Bottom fraction of image to keep (0.4–0.6 recommended)')
    args = parser.parse_args()

    train(args)