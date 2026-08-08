"""Standalone evaluation script."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from models.cnn_model import DigitCNN
from training.dataset_loader import create_dataloaders
from training.metrics import MetricsTracker
from utils import visualizer
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DigitCNN checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth checkpoint.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/dataset",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = DigitCNN()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded checkpoint from '{args.checkpoint}' "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"val_acc={checkpoint.get('val_acc', float('nan')):.4f})"
    )

    # DataLoaders
    _, val_loader, _ = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=2,
    )

    # Evaluation loop
    tracker = MetricsTracker(track_logits=True)
    eval_start = time.perf_counter()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            tracker.update(preds, labels_dev, logits)
    eval_elapsed = time.perf_counter() - eval_start
    print(f"Evaluation completed in {eval_elapsed:.2f}s")

    metrics = tracker.compute()
    report = metrics["classification_report"]
    cm = metrics["confusion_matrix"]
    per_class_acc = metrics["per_class_accuracy"]

    # Save classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nClassification report saved -> {report_path}")
    print(report)

    # Print per-class accuracy table with common misclassification
    print(
        f"\n{'Digit':>6} | {'Samples':>7} | {'Accuracy':>8} | Common Misclassification"
    )
    print("-" * 60)
    for cls in range(10):
        row = cm[cls]
        total = int(row.sum())
        correct = int(row[cls])
        acc_pct = per_class_acc[cls] * 100

        # Find most common misclassification
        errors = [(i, row[i]) for i in range(10) if i != cls and row[i] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)
        if errors:
            top_err_cls, top_err_cnt = errors[0]
            err_str = f"→ {top_err_cls} ({top_err_cnt / max(total, 1) * 100:.1f}%)"
        else:
            err_str = "—"

        print(f"{cls:>6}   | {total:>7} | {acc_pct:>7.1f}% | {err_str}")

    print(f"\nOverall accuracy: {metrics['accuracy'] * 100:.2f}%")
    if "top3_accuracy" in metrics:
        print(f"Top-3 accuracy:   {metrics['top3_accuracy'] * 100:.2f}%")
        print(f"Top-5 accuracy:   {metrics['top5_accuracy'] * 100:.2f}%")

    logger.info(
        "Evaluation: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, kappa=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["cohen_kappa"],
    )

    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "accuracy": metrics["accuracy"],
                "top3_accuracy": metrics.get("top3_accuracy"),
                "top5_accuracy": metrics.get("top5_accuracy"),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "weighted_f1": metrics["weighted_f1"],
                "cohen_kappa": metrics["cohen_kappa"],
                "total_samples": metrics["total_samples"],
                "total_errors": metrics["total_errors"],
                "eval_seconds": round(eval_elapsed, 3),
            },
            f,
            indent=2,
        )
    print(f"Metrics summary saved -> {summary_path}")

    # Save per-class accuracy to CSV for programmatic use
    csv_path = os.path.join(args.output_dir, "per_class_accuracy.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["digit", "samples", "correct", "accuracy_pct", "top_misclassified_as", "misclass_pct"])
        for cls in range(10):
            row = cm[cls]
            total = int(row.sum())
            correct = int(row[cls])
            acc_pct = round(per_class_acc[cls] * 100, 2)
            errors = [(i, row[i]) for i in range(10) if i != cls and row[i] > 0]
            errors.sort(key=lambda x: x[1], reverse=True)
            if errors:
                top_err_cls, top_err_cnt = errors[0]
                misclass_pct = round(top_err_cnt / max(total, 1) * 100, 2)
            else:
                top_err_cls, misclass_pct = "", 0.0
            writer.writerow([cls, total, correct, acc_pct, top_err_cls, misclass_pct])
    print(f"Per-class accuracy CSV saved -> {csv_path}")

    # Confusion matrix plot
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    visualizer.plot_confusion_matrix(model, val_loader, device, save_path=cm_path)

    # Per-class accuracy bar chart
    bar_path = os.path.join(args.output_dir, "per_class_accuracy.png")
    _plot_per_class_bar(per_class_acc, metrics["accuracy"], bar_path)


def _plot_per_class_bar(
    per_class_acc: list[float],
    overall_acc: float,
    save_path: str,
) -> None:
    """Generate a horizontal bar chart of per-class accuracy.

    Bars are color-coded:
      - Green  (≥ 95%) — strong performance
      - Orange (≥ 85%) — acceptable but improvable
      - Red    (< 85%) — needs attention
    """
    digits = list(range(10))
    acc_pcts = [a * 100 for a in per_class_acc]

    # Color-code by threshold
    colors = []
    for a in acc_pcts:
        if a >= 95:
            colors.append("#2ecc71")   # green
        elif a >= 85:
            colors.append("#f39c12")   # orange
        else:
            colors.append("#e74c3c")   # red

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(digits, acc_pcts, color=colors, edgecolor="white", height=0.7)

    # Annotate each bar with its value
    for bar, pct in zip(bars, acc_pcts):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold",
        )

    # Overall accuracy reference line
    overall_pct = overall_acc * 100
    ax.axvline(x=overall_pct, color="#3498db", linestyle="--", linewidth=1.5,
               label=f"Overall: {overall_pct:.1f}%")

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Digit")
    ax.set_yticks(digits)
    ax.set_xlim(0, 105)
    ax.set_title("Per-Class Accuracy")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # digit 0 at top

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Per-class accuracy chart saved -> {save_path}")


if __name__ == "__main__":
    main()
