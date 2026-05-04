"""Split data/raw/ into data/dataset/train/ and data/dataset/val/ (80/20)."""

import os
import shutil
import random
import argparse
import time


def prepare_dataset(
    raw_dir: str = "./data/augmented",
    dataset_dir: str = "./data/dataset",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Split source images into train/val directories.

    Args:
        raw_dir: Path to augmented (or raw) images with class sub-folders.
        dataset_dir: Output root — ``train/`` and ``val/`` sub-folders are
                     created inside this directory.
        train_ratio: Fraction of images allocated to the training set.
        seed: Random seed for reproducible splits.
    """
    if not os.path.isdir(raw_dir):
        print(f"Error: source directory '{raw_dir}' does not exist.")
        return

    random.seed(seed)

    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    total_train = 0
    total_val = 0
    start_time = time.perf_counter()

    classes = sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    )

    print(f"Splitting {len(classes)} classes with {train_ratio:.0%} train ratio (seed={seed})...")

    for class_name in classes:
        class_path = os.path.join(raw_dir, class_name)

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(supported_ext)
        ]
        random.shuffle(images)

        split = int(len(images) * train_ratio)
        train_imgs = images[:split]
        val_imgs = images[split:]

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest_dir = os.path.join(dataset_dir, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img in imgs:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(dest_dir, img),
                )

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        print(f"  Class {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

    elapsed = time.perf_counter() - start_time
    print(f"\nDone! ({elapsed:.1f}s)")
    print(f"  Total train : {total_train}")
    print(f"  Total val   : {total_val}")
    print(f"  Dataset at  : {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split augmented/raw images into train and validation sets.",
    )
    parser.add_argument(
        "--raw_dir", type=str, default="./data/augmented",
        help="Source directory with class sub-folders (default: ./data/augmented).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./data/dataset",
        help="Output dataset directory (default: ./data/dataset).",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Fraction of images for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        dataset_dir=args.out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
