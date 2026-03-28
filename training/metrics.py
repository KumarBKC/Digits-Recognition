"""Accuracy, confusion matrix, per-class accuracy tracking."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


class MetricsTracker:
    """Accumulate predictions across batches and compute epoch-level metrics."""

    def __init__(self):
        self._all_preds: List[int] = []
        self._all_labels: List[int] = []

    # Accumulation

    def update(self, preds_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        """Append batch predictions and ground-truth labels.

        Args:
            preds_tensor: 1-D integer tensor of predicted class indices.
            labels_tensor: 1-D integer tensor of ground-truth labels.
        """
        self._all_preds.extend(preds_tensor.cpu().tolist())
        self._all_labels.extend(labels_tensor.cpu().tolist())

    def reset(self) -> None:
        """Clear accumulated state between epochs."""
        self._all_preds = []
        self._all_labels = []

    # Computation

    def compute(self) -> Dict:
        """Compute all metrics over accumulated predictions.

        Returns:
            Dictionary with keys:
              * ``accuracy`` – overall float accuracy
              * ``per_class_accuracy`` – list[float] indexed by digit
              * ``confusion_matrix`` – 10×10 numpy array
              * ``classification_report`` – sklearn-formatted string
              * ``top1_errors`` – list of ``(true, pred, confidence)``
                tuples (confidence is –1 when unavailable)
        """
        preds = np.array(self._all_preds)
        labels = np.array(self._all_labels)

        overall_acc = float((preds == labels).mean()) if len(labels) > 0 else 0.0

        cm = confusion_matrix(labels, preds, labels=list(range(10)))

        per_class_acc: List[float] = []
        for cls in range(10):
            row_sum = cm[cls].sum()
            acc = float(cm[cls, cls] / row_sum) if row_sum > 0 else 0.0
            per_class_acc.append(acc)

        report = classification_report(
            labels,
            preds,
            labels=list(range(10)),
            target_names=[str(d) for d in range(10)],
            zero_division=0,
        )

        # top1_errors: samples that were misclassified
        top1_errors: List[Tuple[int, int, float]] = [
            (int(labels[i]), int(preds[i]), -1.0)
            for i in range(len(labels))
            if labels[i] != preds[i]
        ]

        return {
            "accuracy": overall_acc,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "top1_errors": top1_errors,
        }
