"""Split data/raw/ into data/dataset/train/ and data/dataset/val/ (80/20)."""

import os
import shutil
import random

RAW_DIR = "./data/augmented"
DATASET_DIR = "./data/dataset"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

for class_name in sorted(os.listdir(RAW_DIR)):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
    random.shuffle(images)

    split = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split]
    val_imgs = images[split:]

    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest_dir = os.path.join(DATASET_DIR, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(dest_dir, img))

    print(f"Class {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\nDone! Dataset ready at data/dataset/")
