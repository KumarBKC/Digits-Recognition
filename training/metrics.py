"""Accuracy, confusion matrix, per-class accuracy tracking."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsTracker:
    """Accumulate predictions across batches and compute epoch-level metrics."""

    def __init__(self):
        self._all_preds: List[int] = []
        self._all_labels: List[int] = []
        self._epoch_start: Optional[float] = None
        self._epoch_elapsed: float = 0.0
        self._batch_count: int = 0

    # Accumulation

    def update(self, preds_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> None:
        """Append batch predictions and ground-truth labels.

        Args:
            preds_tensor: 1-D integer tensor of predicted class indices.
            labels_tensor: 1-D integer tensor of ground-truth labels.
        """
        self._all_preds.extend(preds_tensor.cpu().tolist())
        self._all_labels.extend(labels_tensor.cpu().tolist())
        self._batch_count += 1

    def reset(self) -> None:
        """Clear accumulated state between epochs."""
        self._all_preds = []
        self._all_labels = []
        self._batch_count = 0
        self._epoch_start = time.perf_counter()
        self._epoch_elapsed = 0.0

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
        # Finalise epoch timing
        if self._epoch_start is not None:
            self._epoch_elapsed = time.perf_counter() - self._epoch_start

        preds = np.array(self._all_preds)
        labels = np.array(self._all_labels)

        overall_acc = float((preds == labels).mean()) if len(labels) > 0 else 0.0

        cm = confusion_matrix(labels, preds, labels=list(range(10)))

        per_class_acc: List[float] = []
        for cls in range(10):
            row_sum = cm[cls].sum()
            acc = float(cm[cls, cls] / row_sum) if row_sum > 0 else 0.0
            per_class_acc.append(acc)

        # Macro-averaged precision, recall, F1
        macro_precision = float(precision_score(
            labels, preds, labels=list(range(10)), average="macro", zero_division=0,
        ))
        macro_recall = float(recall_score(
            labels, preds, labels=list(range(10)), average="macro", zero_division=0,
        ))
        macro_f1 = float(f1_score(
            labels, preds, labels=list(range(10)), average="macro", zero_division=0,
        ))

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

        error_rate = 1.0 - overall_acc if len(labels) > 0 else 0.0

        return {
            "accuracy": overall_acc,
            "error_rate": error_rate,
            "per_class_accuracy": per_class_acc,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "top1_errors": top1_errors,
            "total_samples": len(labels),
            "total_errors": len(top1_errors),
            "batches_processed": self._batch_count,
            "elapsed_seconds": round(self._epoch_elapsed, 3),
        }

    def summary(self) -> str:
        """Return a concise human-readable summary of the last compute()."""
        m = self.compute()
        lines = [
            f"Accuracy:  {m['accuracy'] * 100:.2f}%",
            f"Error rate: {m['error_rate'] * 100:.2f}%",
            f"Precision: {m['precision'] * 100:.2f}%",
            f"Recall:    {m['recall'] * 100:.2f}%",
            f"F1 Score:  {m['f1_score'] * 100:.2f}%",
            f"Samples:   {m['total_samples']:,}  ({m['total_errors']} errors)",
            f"Batches:   {m['batches_processed']}",
            f"Time:      {m['elapsed_seconds']:.3f}s",
        ]
        return "\n".join(lines)
