"""Single-image prediction pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from inference.preprocessor import ImagePreprocessor
from models.cnn_model import DigitCNN


@dataclass
class PredictionResult:
    """Result returned by :class:`DigitPredictor`."""

    digit: int
    confidence: float
    all_probs: List[float]
    processing_time_ms: float

    def top_k(self, k: int = 3) -> List[Tuple[int, float]]:
        """Return the *k* highest-confidence (digit, probability) pairs.

        Args:
            k: Number of top predictions to return (default 3).

        Returns:
            Sorted list of ``(digit, probability)`` tuples, highest first.
        """
        indexed = list(enumerate(self.all_probs))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(digit, prob) for digit, prob in indexed[:k]]

    def __repr__(self) -> str:
        return (
            f"PredictionResult(digit={self.digit}, "
            f"confidence={self.confidence:.4f}, "
            f"time={self.processing_time_ms:.1f}ms)"
        )


class DigitPredictor:
    """Load a trained DigitCNN checkpoint and run predictions."""

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.65

    def __init__(self, model_path: str, device: str = "auto"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{model_path}'. "
                "Train a model first with train.py or provide a valid path."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_path = model_path

        self.model = DigitCNN()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._checkpoint_epoch = checkpoint.get("epoch", None)
        self._checkpoint_acc = checkpoint.get("val_acc", None)

        self.preprocessor = ImagePreprocessor(device=str(self.device))

    # Public API

    def preprocess(self, image) -> torch.Tensor:
        """Preprocess any supported input to a model-ready tensor."""
        return self.preprocessor.preprocess_for_inference(image)

    def predict(self, image) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image: ``np.ndarray`` (BGR), ``PIL.Image``, or file path.

        Returns:
            :class:`PredictionResult` with digit, confidence and probabilities.
        """
        t0 = time.perf_counter()

        tensor = self.preprocess(image)  # [1, 1, 43, 17]

        with torch.no_grad():
            probs = self.model.predict_proba(tensor)  # [1, 10]

        probs_list = probs.squeeze(0).cpu().tolist()
        digit = int(torch.argmax(probs).item())
        confidence = float(probs_list[digit])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResult(
            digit=digit,
            confidence=confidence,
            all_probs=probs_list,
            processing_time_ms=elapsed_ms,
        )

    def predict_batch(self, images: list) -> List[PredictionResult]:
        """Run inference on a list of images efficiently.

        Images are stacked into a single batch tensor for GPU efficiency.

        Args:
            images: List of inputs accepted by :meth:`preprocess`.

        Returns:
            List of :class:`PredictionResult`, one per image.
        """
        t0 = time.perf_counter()

        tensors = [self.preprocess(img) for img in images]  # list of [1,1,43,17]
        batch = torch.cat(tensors, dim=0)  # [N, 1, 43, 17]

        with torch.no_grad():
            probs = self.model.predict_proba(batch)  # [N, 10]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_image_ms = elapsed_ms / max(len(images), 1)

        results: List[PredictionResult] = []
        for i in range(len(images)):
            probs_list = probs[i].cpu().tolist()
            digit = int(probs[i].argmax().item())
            results.append(
                PredictionResult(
                    digit=digit,
                    confidence=float(probs_list[digit]),
                    all_probs=probs_list,
                    processing_time_ms=per_image_ms,
                )
            )
        return results

    def predict_or_reject(
        self, image, threshold: Optional[float] = None,
    ) -> Optional[PredictionResult]:
        """Predict a digit, returning ``None`` if confidence is too low.

        This is a convenience wrapper around :meth:`predict` and
        :meth:`is_confident` for pipelines that discard uncertain outputs.

        Args:
            image: Any input accepted by :meth:`predict`.
            threshold: Minimum confidence to accept (defaults to
                       :attr:`DEFAULT_CONFIDENCE_THRESHOLD`).

        Returns:
            :class:`PredictionResult` if confident, otherwise ``None``.
        """
        if threshold is None:
            threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
        result = self.predict(image)
        return result if result.confidence >= threshold else None

    @staticmethod
    def is_confident(result: PredictionResult, threshold: float = 0.65) -> bool:
        """Return True if the prediction confidence exceeds the threshold."""
        return result.confidence >= threshold

    def __repr__(self) -> str:
        epoch_str = f", epoch={self._checkpoint_epoch}" if self._checkpoint_epoch else ""
        acc_str = f", val_acc={self._checkpoint_acc:.4f}" if self._checkpoint_acc else ""
        return (
            f"DigitPredictor(device={self.device}{epoch_str}{acc_str})"
        )
