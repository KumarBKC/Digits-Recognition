"""Standalone evaluation script."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

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
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    tracker = MetricsTracker()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            tracker.update(preds, labels_dev)

    metrics = tracker.compute()
    report = metrics["classification_report"]
    cm = metrics["confusion_matrix"]
    per_class_acc = metrics["per_class_accuracy"]

    # Save classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nClassification report saved → {report_path}")
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

    # Confusion matrix plot
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    visualizer.plot_confusion_matrix(model, val_loader, device, save_path=cm_path)


if __name__ == "__main__":
    main()
