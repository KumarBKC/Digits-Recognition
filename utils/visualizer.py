"""Visualisation utilities — training curves, confusion matrix, sample predictions."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader


def plot_history(history: Dict[str, List[float]], save_path: str = "training_curves.png") -> None:
    """Plot train / val loss and accuracy curves.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
        save_path: Where to save the figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Epoch of best validation accuracy
    best_idx = int(np.argmax(history["val_acc"]))
    best_epoch = best_idx + 1
    best_val_acc_pct = history["val_acc"][best_idx] * 100

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train acc")
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val acc")
    ax_acc.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy Curves")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.suptitle(
        f"Training History — best val acc {best_val_acc_pct:.1f}% @ epoch {best_epoch}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves -> {save_path}")


def plot_confusion_matrix(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str = "confusion_matrix.png",
) -> None:
    """Compute and plot a normalised 10×10 confusion matrix.

    Args:
        model: Trained DigitCNN.
        val_loader: Validation DataLoader.
        device: Torch device.
        save_path: Output PNG path.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = sk_cm(all_labels, all_preds, labels=list(range(10)))
    # Normalise row-wise (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix -> {save_path}")

