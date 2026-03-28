"""Image preprocessing pipeline for inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class ImagePreprocessor:
    """Deterministic preprocessing pipeline that prepares any image for the model.

    Every image entering the model must pass through an identical pipeline so
    that training distribution matches inference distribution.
    """

    def __init__(self, device: str = "cpu", debug: bool = False):
        self.device = torch.device(device)
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_for_inference(self, image) -> torch.Tensor:
        """Convert any supported input to a model-ready tensor.

        Args:
            image: One of:
                * ``numpy.ndarray`` — OpenCV BGR or grayscale image
                * ``PIL.Image.Image`` — any mode
                * ``str`` — file path

        Returns:
            Float32 tensor of shape ``[1, 1, 43, 17]`` on ``self.device``.
        """
        # Step 1 — Input normalisation
        gray = self._to_gray(image)

        # Step 2 — Background detection / inversion
        gray = self._normalize_background(gray)

        # Step 3 — Crop to digit bounding box
        gray = self._crop_to_digit(gray)

        # Step 4 — Resize to (17, 43)  [width × height for cv2]
        resized = cv2.resize(gray, (17, 43), interpolation=cv2.INTER_AREA)

        # Step 5 — Normalise pixel values to [-1, 1]
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5

        # Step 6 — Tensor conversion → [1, 1, 43, 17]
        arr = np.expand_dims(arr, axis=0)  # [1, 43, 17]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 1, 43, 17]
        tensor = tensor.to(self.device)

        # Step 7 — Sanity checks (debug mode)
        if self.debug:
            assert tensor.shape == (1, 1, 43, 17), f"Bad shape: {tensor.shape}"
            assert tensor.dtype == torch.float32, f"Bad dtype: {tensor.dtype}"
            assert tensor.min() >= -1.1 and tensor.max() <= 1.1, (
                f"Values out of range: [{tensor.min():.2f}, {tensor.max():.2f}]"
            )

        return tensor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image) -> np.ndarray:
        """Convert any supported input to a uint8 grayscale ndarray."""
        # File path
        if isinstance(image, str):
            from PIL import Image as PILImage

            image = PILImage.open(image).convert("L")

        # PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                return np.array(image.convert("L"), dtype=np.uint8)
        except ImportError:
            pass

        # NumPy / OpenCV
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            if image.ndim == 2:
                return image.astype(np.uint8)

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _normalize_background(gray: np.ndarray) -> np.ndarray:
        """Invert image if background is dark (assume ink-on-white convention)."""
        # Sample border region (5px from each edge)
        h, w = gray.shape
        border_pixels = np.concatenate(
            [
                gray[:5, :].ravel(),
                gray[-5:, :].ravel(),
                gray[:, :5].ravel(),
                gray[:, -5:].ravel(),
            ]
        )
        mean_border = float(border_pixels.mean()) if len(border_pixels) > 0 else 128.0

        if mean_border < 128:  # dark background → invert
            return cv2.bitwise_not(gray)
        return gray

    @staticmethod
    def _crop_to_digit(gray: np.ndarray) -> np.ndarray:
        """Crop to digit bounding box, removing surrounding whitespace."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)

        if coords is None:
            return gray  # nothing found — return as-is

        x, y, w, h = cv2.boundingRect(coords)
        pad = 4
        h_img, w_img = gray.shape
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)

        cropped = gray[y0:y1, x0:x1]
        if cropped.size == 0:
            return gray
        return cropped
