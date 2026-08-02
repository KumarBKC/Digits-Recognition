"""Generate augmented images using rotation, brightness jitter, and elastic
deformation to increase dataset diversity."""

import os
import time
import random
import argparse

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def elastic_deform(
    img: Image.Image,
    alpha: float = 8.0,
    sigma: float = 3.0,
) -> Image.Image:
    """Apply elastic deformation to a PIL Image.

    Generates smooth random displacement fields and warps the image,
    simulating natural handwriting variation.

    Args:
        img: Grayscale PIL Image.
        alpha: Displacement intensity — higher values produce stronger warps.
        sigma: Gaussian smoothing kernel for the displacement field.

    Returns:
        Elastically deformed PIL Image.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random displacement fields smoothed by a Gaussian
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode="constant", cval=0)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode="constant", cval=0)

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    indices = [np.clip(y_coords + dy, 0, h - 1), np.clip(x_coords + dx, 0, w - 1)]

    deformed = map_coordinates(arr, indices, order=1, mode="reflect")
    return Image.fromarray(np.clip(deformed, 0, 255).astype(np.uint8))


def augment_dataset(
    raw_dir: str,
    augmented_dir: str,
    rotations_count: int = 50,
    brightness_jitter: float = 0.0,
    elastic: bool = False,
    elastic_alpha: float = 8.0,
    elastic_sigma: float = 3.0,
    seed: int | None = None,
    output_format: str = "png",
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
        elastic: If True, apply elastic deformation to each augmented image.
        elastic_alpha: Displacement intensity for elastic deformation.
        elastic_sigma: Gaussian smoothing sigma for elastic deformation.
        seed: Optional random seed for reproducible brightness jitter.
    """
    if rotations_count < 1:
        print("Error: rotations_count must be at least 1.")
        return
    if not 0.0 <= brightness_jitter <= 1.0:
        print("Error: brightness_jitter must be between 0.0 and 1.0.")
        return

    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' not found.")
        return

    os.makedirs(augmented_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    # Calculate degree range
    half_rot = rotations_count // 2
    degrees = range(-half_rot, rotations_count - half_rot)

    # Process each class (0-9)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not classes:
        print(f"Error: No class sub-folders found in '{raw_dir}'.")
        return

    total_images_processed = 0
    total_augmented_saved = 0
    skipped = 0
    start_time = time.perf_counter()

    jitter_str = f", brightness jitter ±{brightness_jitter*100:.0f}%" if brightness_jitter > 0 else ""
    elastic_str = f", elastic deform (α={elastic_alpha}, σ={elastic_sigma})" if elastic else ""
    print(f"Starting augmentation: {rotations_count} rotations per image{jitter_str}{elastic_str}...")

    for class_name in sorted(classes):
        src_class_path = os.path.join(raw_dir, class_name)
        dst_class_path = os.path.join(augmented_dir, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        images = [
            f for f in os.listdir(src_class_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not images:
            print(f"  Skipping class {class_name}: no images found.")
            continue

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

                        # Optional elastic deformation
                        if elastic:
                            rotated = elastic_deform(rotated, alpha=elastic_alpha, sigma=elastic_sigma)

                        # Save augmented version
                        save_name = f"{base_name}_rot{deg}.{output_format.lstrip('.') }"
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
    parser = argparse.ArgumentParser(description="Augment digit dataset by rotation, brightness jitter, and elastic deformation.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw source images.")
    parser.add_argument("--out_dir", type=str, default="data/augmented", help="Output path for augmented images.")
    parser.add_argument("--count", type=int, default=50, help="Number of rotations per image.")
    parser.add_argument(
        "--brightness_jitter", type=float, default=0.0,
        help="Random brightness jitter fraction, e.g. 0.15 for ±15%%. (default: 0 = disabled)",
    )
    parser.add_argument(
        "--elastic", action="store_true",
        help="Apply elastic deformation to each augmented image.",
    )
    parser.add_argument(
        "--elastic_alpha", type=float, default=8.0,
        help="Elastic deformation displacement intensity (default: 8.0).",
    )
    parser.add_argument(
        "--elastic_sigma", type=float, default=3.0,
        help="Elastic deformation Gaussian smoothing sigma (default: 3.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible jitter.")
    parser.add_argument("--format", type=str, default="png", help="Output image format (png, jpeg, etc.).")

    args = parser.parse_args()
    augment_dataset(
        args.raw_dir,
        args.out_dir,
        args.count,
        brightness_jitter=args.brightness_jitter,
        elastic=args.elastic,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
        seed=args.seed,
        output_format=args.format,
    )
