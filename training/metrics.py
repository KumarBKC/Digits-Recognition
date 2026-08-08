"""Accuracy, confusion matrix, per-class accuracy tracking.

Extended metrics include top-K accuracy, weighted-average scores,
and Cohen's Kappa coefficient for a comprehensive evaluation.
"""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple


import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsTracker:
    """Accumulate predictions across batches and compute epoch-level metrics.

    Optionally stores raw logits so that top-K accuracy can be computed
    without re-running inference.
    """

    def __init__(self, track_logits: bool = False):
        self._all_preds: List[int] = []
        self._all_labels: List[int] = []
        self._track_logits = track_logits
        self._all_logits: List[np.ndarray] = []

    # Accumulation

    def update(
        self,
        preds_tensor: torch.Tensor,
        labels_tensor: torch.Tensor,
        logits_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Append batch predictions and ground-truth labels.

        Args:
            preds_tensor: 1-D integer tensor of predicted class indices.
            labels_tensor: 1-D integer tensor of ground-truth labels.
            logits_tensor: Optional raw logits ``[B, C]`` — needed for
                top-K accuracy computation.
        """
        self._all_preds.extend(preds_tensor.cpu().tolist())
        self._all_labels.extend(labels_tensor.cpu().tolist())
        if self._track_logits and logits_tensor is not None:
            self._all_logits.append(logits_tensor.detach().cpu().numpy())

    def reset(self) -> None:
        """Clear accumulated state between epochs."""
        self._all_preds = []
        self._all_labels = []
        self._all_logits = []

    # ---- Top-K accuracy ---------------------------------------------------

    def top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-K accuracy from stored logits.

        Returns 0.0 if logits were not tracked.
        """
        if not self._all_logits:
            return 0.0
        logits = np.concatenate(self._all_logits, axis=0)  # [N, C]
        labels = np.array(self._all_labels)
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]   # [N, k]
        correct = np.any(top_k_preds == labels[:, None], axis=1)
        return float(correct.mean())

    # Computation

    def compute(self) -> Dict:
        """Compute all metrics over accumulated predictions.

        Returns:
            Dictionary with keys:
              * ``accuracy`` – overall float accuracy
              * ``error_rate`` – 1 − accuracy
              * ``per_class_accuracy`` – list[float] indexed by digit
              * ``precision`` / ``recall`` / ``f1_score`` – macro-averaged
              * ``weighted_precision`` / ``weighted_recall`` / ``weighted_f1``
              * ``cohen_kappa`` – Cohen's Kappa coefficient
              * ``top3_accuracy`` / ``top5_accuracy`` – (if logits tracked)
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

        _metric_kwargs = dict(labels=list(range(10)), zero_division=0)

        # Macro-averaged precision, recall, F1
        macro_precision = float(precision_score(labels, preds, average="macro", **_metric_kwargs))
        macro_recall = float(recall_score(labels, preds, average="macro", **_metric_kwargs))
        macro_f1 = float(f1_score(labels, preds, average="macro", **_metric_kwargs))

        # Weighted-averaged (accounts for class imbalance)
        weighted_precision = float(precision_score(labels, preds, average="weighted", **_metric_kwargs))
        weighted_recall = float(recall_score(labels, preds, average="weighted", **_metric_kwargs))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", **_metric_kwargs))

        # Cohen's Kappa — agreement beyond chance
        kappa = float(cohen_kappa_score(labels, preds)) if len(labels) > 0 else 0.0

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

        result: Dict = {
            "accuracy": overall_acc,
            "error_rate": error_rate,
            "per_class_accuracy": per_class_acc,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "cohen_kappa": kappa,
            "confusion_matrix": cm,
            "classification_report": report,
            "top1_errors": top1_errors,
            "total_samples": len(labels),
            "total_errors": len(top1_errors),
        }

        # Top-K accuracy (only if logits were stored)
        if self._all_logits:
            result["top3_accuracy"] = self.top_k_accuracy(3)
            result["top5_accuracy"] = self.top_k_accuracy(5)

        return result

    def summary(self) -> str:
        """Return a concise human-readable summary of the last compute()."""
        m = self.compute()
        lines = [
            f"Accuracy:  {m['accuracy'] * 100:.2f}%",
            f"Error rate: {m['error_rate'] * 100:.2f}%",
            f"Precision: {m['precision'] * 100:.2f}%  (macro)  |  {m['weighted_precision'] * 100:.2f}%  (weighted)",
            f"Recall:    {m['recall'] * 100:.2f}%  (macro)  |  {m['weighted_recall'] * 100:.2f}%  (weighted)",
            f"F1 Score:  {m['f1_score'] * 100:.2f}%  (macro)  |  {m['weighted_f1'] * 100:.2f}%  (weighted)",
            f"Kappa:     {m['cohen_kappa']:.4f}",
        ]
        if "top3_accuracy" in m:
            lines.append(f"Top-3 Acc: {m['top3_accuracy'] * 100:.2f}%")
            lines.append(f"Top-5 Acc: {m['top5_accuracy'] * 100:.2f}%")
        lines.append(f"Samples:   {m['total_samples']:,}  ({m['total_errors']} errors)")
        return "\n".join(lines)
