"""Single-image prediction pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass,field
from typing import List

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


class DigitPredictor:
    """Load a trained DigitCNN checkpoint and run predictions."""

    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = DigitCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

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

    @staticmethod
    def is_confident(result: PredictionResult, threshold: float = 0.65) -> bool:
        """Return True if the prediction confidence exceeds the threshold."""
        return result.confidence >= threshold
