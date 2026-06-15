#!/usr/bin/env python3
"""Visualize transformed images from the dataset.

Shows original PIL images and transformed tensor images side-by-side.
"""
import argparse
import os
import sys

import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import dataset and transforms from dualhead
from dualhead import CSVRegressionDataset, BottomCenterCrop, CLAHETransform


def denormalize(tensor):
    """Denormalize ImageNet-normalized tensor back to [0, 1] range."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def visualize_dataset(args):
    """Load and display transformed images from the dataset."""
    
    # Prepare CLAHE transform (no-op if OpenCV not available)
    if CLAHETransform is not None:
        clahe = CLAHETransform(clip_limit=args.clahe_clip, tile_grid_size=(args.clahe_grid, args.clahe_grid))
    else:
        clahe = T.Lambda(lambda x: x)

    # Transform pipeline (same as training)
    train_transform = T.Compose([
        T.Resize(256),
        BottomHalfResize(fraction=0.5, size=224),
        T.RandomRotation(degrees=5),          # small rotations
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=1, saturation=0.2, hue=0.2),
        ], p=0.5),
        clahe,
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5))
        ], p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = CSVRegressionDataset(
        root_dir=args.data_dir,
        transform=transform,
        img_type='RGB',
        flip_augment=False,  # disable flipping for visualization
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Create figure
    num_samples = min(args.num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        # Load raw image
        img_path, targets = dataset.data_samples[idx]
        raw_img = Image.open(img_path).convert('RGB')
        
        # Get transformed image
        transformed_img, _ = dataset[idx]
        
        # Denormalize for display
        transformed_display = denormalize(transformed_img).permute(1, 2, 0).numpy()
        
        # Plot original
        axes[idx, 0].imshow(raw_img)
        axes[idx, 0].set_title(f"Original\nPath: {os.path.basename(img_path)}")
        axes[idx, 0].axis('off')
        
        # Plot transformed
        axes[idx, 1].imshow(transformed_display)
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        axes[idx, 1].set_title(
            f"Transformed (BottomCenterCrop + CLAHE + Aug)\n"
            f"val1={targets_np[0]:.3f}, val2={targets_np[1]:.3f}"
        )
        axes[idx, 1].axis('off')

    plt.tight_layout()
    
    # Save and/or show
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {args.output}")
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize transformed images from dataset')
    parser.add_argument('--data-dir', type=str, default='put_jetbot_dataset/dataset/train',
                        help='Path to train folder')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='./tmp/dataset_transforms.png',
                        help='Path to save visualization image (default: /tmp/dataset_transforms.png)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Display the plot (headless environments: use --output instead)')
    parser.add_argument('--clahe-clip', type=float, default=5.0, help='CLAHE clip limit')
    parser.add_argument('--clahe-grid', type=int, default=15, help='CLAHE tile grid size')
    args = parser.parse_args()

    if not args.output and not args.show:
        print("Note: Running in headless mode. Saving to /tmp/dataset_transforms.png by default.")
        args.output = '/tmp/dataset_transforms.png'

    visualize_dataset(args)
