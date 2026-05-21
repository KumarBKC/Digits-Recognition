"""Custom Dataset and DataLoader factory for digit recognition."""

from __future__ import annotations

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.augmentation import train_transforms, val_transforms


class DigitDataset(Dataset):

    def __init__(self, root_dir: str, split: str = "train", transform=None,):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                label = int(class_name)
            except ValueError:
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("L")
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            from torchvision import transforms
            img = transforms.ToTensor()(img)
            
        return img, label

    def get_class_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for _, label in self.samples:
            dist[label] = dist.get(label, 0) + 1
        return dist

    def visualize_samples(self, n: int = 5) -> None:
        classes = sorted(set(label for _, label in self.samples))
        fig, axes = plt.subplots(len(classes), n, figsize=(n * 2, len(classes) * 2))

        for row, cls in enumerate(classes):
            cls_samples = [(p, l) for p, l in self.samples if l == cls]
            chosen = random.sample(cls_samples, min(n, len(cls_samples)))
            for col, (path, _) in enumerate(chosen):
                img = Image.open(path).convert("L")
                ax = axes[row][col] if len(classes) > 1 else axes[col]
                ax.imshow(img, cmap="gray")
                ax.set_title(str(cls), fontsize=8)
                ax.axis("off")

        plt.suptitle(f"Sample images - {self.split} split")
        plt.tight_layout()
        plt.savefig(f"samples_{self.split}.png", dpi=100)
        print(f"Saved samples_{self.split}.png")
        plt.show()

    def validate_integrity(self, verbose: bool = True) -> Dict[str, int]:
        """Check every sample file for readability and report issues.

        Scans each image in the dataset to verify it can be opened
        without errors.  Corrupt or unreadable files are reported
        but **not** removed from the sample list automatically.

        Args:
            verbose: If True, print progress and per-file errors.

        Returns:
            Dictionary with ``total``, ``valid``, and ``corrupt`` counts.
        """
        valid = 0
        corrupt = 0
        for path, label in self.samples:
            try:
                with open(path, "rb") as f:
                    img = Image.open(f)
                    img.verify()  # checks file integrity without full decode
                valid += 1
            except Exception as exc:
                corrupt += 1
                if verbose:
                    warnings.warn(f"Corrupt image [{label}]: {path} — {exc}")
        stats = {"total": len(self.samples), "valid": valid, "corrupt": corrupt}
        if verbose:
            print(
                f"[{self.split}] Integrity check: {valid}/{len(self.samples)} valid, "
                f"{corrupt} corrupt"
            )
        return stats

    def get_sample_by_class(self, digit: int, n: int = 1) -> List[Tuple[str, int]]:
        """Return up to *n* random samples for a specific digit class.

        Args:
            digit: Target digit (0-9).
            n: Maximum number of samples to return.

        Returns:
            List of ``(file_path, label)`` tuples.
        """
        cls_samples = [(p, l) for p, l in self.samples if l == digit]
        if not cls_samples:
            return []
        return random.sample(cls_samples, min(n, len(cls_samples)))


def create_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 2,) -> Tuple[DataLoader, DataLoader, torch.Tensor]:

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = DigitDataset(train_dir, split="train", transform=train_transforms)
    val_dataset = DigitDataset(val_dir, split="val", transform=val_transforms)

    # --- Compute class weights (inverse frequency) ---
    dist = train_dataset.get_class_distribution()
    num_classes = 10
    class_counts = torch.zeros(num_classes)
    for cls, count in dist.items():
        class_counts[cls] = count

    # Avoid division by zero for missing classes
    class_counts = class_counts.clamp(min=1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes  # normalise

    # --- Weighted sampler so each mini-batch is balanced ---
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


def print_dataset_summary(
    train_dataset: DigitDataset,
    val_dataset: DigitDataset,
    class_weights: torch.Tensor,
) -> None:
    """Print a formatted summary of dataset splits and class weights.

    Args:
        train_dataset: Training split dataset.
        val_dataset: Validation split dataset.
        class_weights: Per-class weight tensor.
    """
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"{'Digit':>6} | {'Train':>7} | {'Val':>7} | {'Weight':>8}")
    print(sep)
    for cls in range(10):
        t_count = train_dist.get(cls, 0)
        v_count = val_dist.get(cls, 0)
        w = float(class_weights[cls])
        print(f"{cls:>6} | {t_count:>7} | {v_count:>7} | {w:>8.4f}")
    print(sep)
    print(f"{'Total':>6} | {len(train_dataset):>7} | {len(val_dataset):>7} |")
    print(f"{sep}\n")


# CLI entry point for validation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset structure")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--data_root", type=str, default="./data/dataset")
    args = parser.parse_args()

    if args.validate:
        for split in ("train", "val"):
            split_dir = os.path.join(args.data_root, split)
            if not os.path.isdir(split_dir):
                print(f"[WARN] {split_dir} does not exist")
                continue
            ds = DigitDataset(split_dir, split=split)
            dist = ds.get_class_distribution()
            print(f"\n=== {split.upper()} split - {len(ds)} samples ===")
            for cls in sorted(dist):
                print(f"  Digit {cls}: {dist[cls]:4d} samples")
