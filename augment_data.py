"""Generate augmented images using rotation and brightness jitter to increase dataset diversity."""

import os
import time
import random
import argparse

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

def augment_dataset(
    raw_dir: str,
    augmented_dir: str,
    rotations_count: int = 50,
    brightness_jitter: float = 0.0,
    seed: int | None = None,
):
    """
    Rotate each image in raw_dir and save to augmented_dir.
    Range is +/- 25 degrees if rotations_count is 50.

    Args:
        raw_dir: Path to directory with class sub-folders of source images.
        augmented_dir: Output directory (class sub-folders created automatically).
        rotations_count: Total number of rotation variants per source image.
        brightness_jitter: If > 0, randomly adjust brightness by ±this fraction
                          (e.g. 0.15 means ±15%). Applied independently to each
                          rotated image.
        seed: Optional random seed for reproducible brightness jitter.
    """
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' not found.")
        return

    os.makedirs(augmented_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate degree range
    half_rot = rotations_count // 2
    degrees = range(-half_rot, rotations_count - half_rot)

    # Process each class (0-9)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    total_images_processed = 0
    total_augmented_saved = 0
    skipped = 0
    start_time = time.perf_counter()

    jitter_str = f", brightness jitter ±{brightness_jitter*100:.0f}%" if brightness_jitter > 0 else ""
    print(f"Starting augmentation: {rotations_count} rotations per image{jitter_str}...")

    for class_name in sorted(classes):
        src_class_path = os.path.join(raw_dir, class_name)
        dst_class_path = os.path.join(augmented_dir, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        images = [
            f for f in os.listdir(src_class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        for img_name in tqdm(images, desc=f"Class {class_name}"):
            img_path = os.path.join(src_class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    base_name = os.path.splitext(img_name)[0]

                    # Determine fill color from top-left pixel
                    fill_color = img.getpixel((0, 0))

                    for deg in degrees:
                        # Rotate image
                        rotated = img.rotate(
                            deg,
                            resample=Image.Resampling.BICUBIC,
                            expand=False,
                            fillcolor=fill_color,
                        )

                        # Optional brightness jitter
                        if brightness_jitter > 0:
                            factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
                            rotated = ImageEnhance.Brightness(rotated).enhance(factor)

                        # Save augmented version
                        save_name = f"{base_name}_rot{deg}.png"
                        rotated.save(os.path.join(dst_class_path, save_name))
                        total_augmented_saved += 1

                total_images_processed += 1
            except Exception as e:
                skipped += 1
                print(f"Failed to process {img_name}: {e}")

    elapsed = time.perf_counter() - start_time
    print(f"\nAugmentation complete! ({elapsed:.1f}s)")
    print(f"  Processed : {total_images_processed} source images")
    print(f"  Generated : {total_augmented_saved} augmented images")
    if skipped:
        print(f"  Skipped   : {skipped} images (errors)")
    print(f"  Saved to  : {augmented_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment digit dataset by rotation and brightness jitter.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw source images.")
    parser.add_argument("--out_dir", type=str, default="data/augmented", help="Output path for augmented images.")
    parser.add_argument("--count", type=int, default=50, help="Number of rotations per image.")
    parser.add_argument(
        "--brightness_jitter", type=float, default=0.0,
        help="Random brightness jitter fraction, e.g. 0.15 for ±15%%. (default: 0 = disabled)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible jitter.")

    args = parser.parse_args()
    augment_dataset(
        args.raw_dir,
        args.out_dir,
        args.count,
        brightness_jitter=args.brightness_jitter,
        seed=args.seed,
    )
