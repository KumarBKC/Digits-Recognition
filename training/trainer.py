"""Training loop, validation, early stopping, and checkpointing."""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Encapsulates the training and validation loop for DigitCNN."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        patience: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch_num: int) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            ``(mean_loss, accuracy)`` for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_num}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple[float, float]:
        """Run a full validation pass.

        Returns:
            ``(mean_loss, accuracy)``
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += images.size(0)

        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train for *num_epochs* epochs with early stopping.

        Returns:
            ``history`` dict with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_acc = 0.0
        epochs_without_improvement = 0
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # LR scheduling
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %03d | train_loss=%.4f  train_acc=%.4f | "
                "val_loss=%.4f  val_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            # Checkpoint on improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                logger.info("  ✓ Saved checkpoint (val_acc=%.4f)", best_val_acc)
                print(f"  ✓ Checkpoint saved (val_acc={best_val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        self.patience,
                    )
                    print(
                        f"Early stopping after {self.patience} epochs "
                        "without improvement."
                    )
                    break

        return history

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> dict:
        """Restore model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(
            "Loaded checkpoint from '%s' (epoch=%d, val_acc=%.4f)",
            path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_acc", float("nan")),
        )
        return checkpoint
