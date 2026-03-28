"""Augmentation transforms for training the digit recognition model."""

import random

import torch
from torchvision import transforms


class AddGaussianNoise:
    """Add Gaussian noise to a tensor with a given probability."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.4):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"
        )


class InvertBrightness:
    """Randomly invert image brightness (white-on-black ↔ black-on-white)."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return 1.0 - tensor  # assumes tensor in [0, 1] before Normalize
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# Training transforms — applied only to the train split
train_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((43, 17)),
        # Geometric augmentations
        transforms.RandomRotation(degrees=12, fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            shear=5,
            fill=255,
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.3,
            fill=255,
        ),
        # Pixel-level augmentations
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.3,
        ),
        transforms.ToTensor(),  # → [0, 1] float32
        InvertBrightness(p=0.3),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # → [-1, 1]
        # Noise augmentation (applied post-tensor)
        AddGaussianNoise(mean=0.0, std=0.05, p=0.4),
    ]
)

# Validation / Test transforms — deterministic, no randomness
val_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((43, 17)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


if __name__ == "__main__":
    import argparse
    import os
    import random as rnd
    import sys

    import matplotlib.pyplot as plt
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="./data/dataset/train")
    args = parser.parse_args()

    if args.preview:
        # Collect all image paths
        image_paths = []
        for root, _, files in os.walk(args.data_root):
            for f in files:
                if f.lower().endswith(".png"):
                    image_paths.append(os.path.join(root, f))

        if not image_paths:
            print("No images found in", args.data_root)
            sys.exit(1)

        sample_path = rnd.choice(image_paths)
        img = Image.open(sample_path).convert("L")

        fig, axes = plt.subplots(1, args.n + 1, figsize=(3 * (args.n + 1), 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        for i in range(1, args.n + 1):
            aug_tensor = train_transforms(img)
            aug_img = aug_tensor.squeeze().numpy()
            axes[i].imshow(aug_img, cmap="gray")
            axes[i].set_title(f"Aug {i}")
            axes[i].axis("off")

        plt.suptitle(f"Augmentation preview: {os.path.basename(sample_path)}")
        plt.tight_layout()
        plt.savefig("augmentation_preview.png", dpi=100)
        print("Saved augmentation_preview.png")
        plt.show()
