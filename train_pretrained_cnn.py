#!/usr/bin/env python3
"""Simple transfer-learning training script using pretrained ResNet18.
Supports classification (ImageFolder) or regression (CSV with image,label).
"""
import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import transforms, datasets
from models.pretrained_cnn import get_resnet18
import tqdm
import pandas as pd

from torch.utils.data import WeightedRandomSampler



class CSVRegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, type='RGB', flip_augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data_samples = []
        self.type = type
        self.flip_augment = flip_augment  # only True for train
        self.flipper = RandomHorizontalFlipWithLabel(p=0.5)

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            
            if os.path.isdir(folder_path):
                csv_path = os.path.join(root_dir, f"{folder_name}.csv")
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    for _, row in df.iterrows():
                        img_name = row.iloc[0]
                        val1 = float(row.iloc[1])
                        val2 = float(row.iloc[2])
                        img_raw = str(row.iloc[0])
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
                else:
                    print(f"Warning: Missing CSV file for folder {folder_name} at {csv_path}")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        img_path, targets = self.data_samples[idx]
        image = Image.open(img_path).convert(self.type)
        targets = torch.tensor(targets, dtype=torch.float32)

        # flip BEFORE the tensor transform pipeline
        if self.flip_augment:
            image, targets = self.flipper(image, targets)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image.resize((224, 224)))

        return image, targets
    

def make_weighted_sampler(dataset: CSVRegressionDataset, bins: int = 10) -> WeightedRandomSampler:
    """
    Oversample based on |val2| so the loader sees more ±1 samples per epoch.
    Bins the val2 range into `bins` equal slices and weights inversely by bin frequency.
    """
    val2s = torch.tensor([s[1][1] for s in dataset.data_samples])  # (N,)
    
    # Bin by magnitude: 0..1 → bin index 0..bins-1
    mag = val2s.abs().clamp(0, 1 - 1e-6)
    bin_idx = (mag * bins).long()  # each sample → a bin

    bin_counts = torch.bincount(bin_idx, minlength=bins).float()
    bin_counts = bin_counts.clamp(min=1)
    bin_weights = 1.0 / bin_counts           # rare bins get high weight
    sample_weights = bin_weights[bin_idx]    # per-sample weight

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

def run_evaluation(model, loader, criterion, device, dataset_len, epoch=None):
    """Evaluate model on a dataloader and return avg MSE loss."""
    model.eval()
    running_loss = 0.0
    num_samples = 0
    label = f'Epoch {epoch} eval' if epoch is not None else 'Evaluating'

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(loader, desc=label, leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            # Verify output shape is (batch, 2)
            assert outputs.shape[1] == 2, (
                f"Expected 2 output values per sample, got {outputs.shape[1]}"
            )

            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)

    avg_loss = running_loss / num_samples

    # Print a few sample predictions from the last batch
    print(f"\n  {'Pred[0]':>10} {'Pred[1]':>10} {'GT[0]':>10} {'GT[1]':>10}")
    for i in range(min(5, outputs.size(0))):
        pred = outputs[i].cpu().tolist()
        gt   = labels[i].cpu().tolist()
        print(f"  {pred[0]:>10.4f} {pred[1]:>10.4f} {gt[0]:>10.4f} {gt[1]:>10.4f}")

    return avg_loss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = T.Compose([
        T.Resize(256),                          # slightly larger before crop
        T.RandomCrop(224),                      # replaces CenterCrop — spatial jitter
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,                           # subtle — don't shift colors too far
        ),         # only if left/right are symmetric in your task
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # simulates focus/motion blur
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # occlusion robustness
    ])

    # test transform stays clean — no augmentation
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # --- Train dataset ---
    train_dataset = CSVRegressionDataset(root_dir=args.data_dir, transform=train_transform, type='RGB', flip_augment=True)
    num_outputs = 2
    print(f'Batch size: {args.batch_size}  |  Train samples: {len(train_dataset)}')

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=args.num_workers,
    )

    # --- Test dataset (adjacent 'test' dir) ---
    test_dir = os.path.join(os.path.dirname(args.data_dir.rstrip('/\\')), 'test')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found at '{test_dir}'. "
            f"Expected it adjacent to '{args.data_dir}'."
        )
    test_dataset = CSVRegressionDataset(root_dir=test_dir, transform=test_transform, type='RGB', flip_augment=False)
    print(f'Test samples: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_resnet18(pretrained=True, num_outputs=num_outputs)
    model = model.to(device)

    criterion = Val2WeightedMSELoss(strategy='magnitude', alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        loop = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=running_loss / ((loop.n + 1) * args.batch_size))

        train_loss = running_loss / len(train_dataset)

        # --- Evaluate on test set ---
        test_loss = run_evaluation(model, test_loader, criterion, device, len(test_dataset), epoch=epoch)

        print(f'Epoch {epoch:>3}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}'
              + (' ← best' if test_loss < best_test_loss else ''))

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if args.checkpoint:
            ckpt_path = os.path.join(args.checkpoint, f'resnet18_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, ckpt_path)

    print(f'\nTraining complete. Best test MSE: {best_test_loss:.4f}')

    if args.output:
        torch.save(model.state_dict(), args.output)

class RandomHorizontalFlipWithLabel:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, targets):
        if torch.rand(1) < self.p:
            image = T.functional.hflip(image)
            targets = targets * torch.tensor([1.0, -1.0])  # negate both steering values
        return image, targets

class Val2WeightedMSELoss(nn.Module):
    """
    MSE loss where each sample is weighted by a function of its val2 label.
    
    Two strategies (pick one):
      'magnitude'  : w = |val2|^alpha  — extremes ±1 get full weight, 0 gets zero
      'suppress'   : w = 1 - exp(-k * val2²) — flat suppression of near-zero, full weight at ±1
    """
    def __init__(self, strategy: str = 'magnitude', alpha: float = 0.5, k: float = 4.0):
        super().__init__()
        assert strategy in ('magnitude', 'suppress')
        self.strategy = strategy
        self.alpha = alpha
        self.k = k

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 2)  — col 0 = val1, col 1 = val2
        val2 = target[:, 1]  # (B,)

        w = 0.4 + 0.6 * val2.abs().pow(0.5)


        # val1 uses unweighted MSE, val2 uses weighted MSE
        loss_val1 = (pred[:, 0] - target[:, 0]).pow(2).mean()
        loss_val2 = ((w*(pred[:, 1] - target[:, 1])).pow(2)).mean()
        return loss_val1 + loss_val2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='put_jetbot_dataset/dataset/train',
                        help='Path to train folder; test folder is inferred as the adjacent "test" dir')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output', type=str, help='Path to save final model state dict')
    parser.add_argument('--checkpoint', type=str, help='Directory to save epoch checkpoints')
    args = parser.parse_args()

    train(args)