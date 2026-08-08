"""Training loop, validation, early stopping, and checkpointing.

Supports:
  - Mixup data augmentation for improved generalization
  - Per-batch LR scheduling (OneCycleLR) or per-epoch (ReduceLROnPlateau)
  - Gradient clipping to prevent exploding gradients
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
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
        mixup_alpha: float = 0.0,
        step_scheduler_per_batch: bool = False,
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
        self.max_grad_norm = 5.0
        self.mixup_alpha = mixup_alpha
        self.step_scheduler_per_batch = step_scheduler_per_batch

        os.makedirs(checkpoint_dir, exist_ok=True)



    def _mixup_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup: interpolate between random pairs of examples.

        Returns:
            ``(mixed_x, y_a, y_b, lam)`` where ``lam`` is the mixing
            coefficient sampled from Beta(alpha, alpha).
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam



    def train_one_epoch(self, epoch_num: int) -> tuple[float, float]:
        """Run one training epoch with optional Mixup augmentation.

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

            # Mixup augmentation (blends pairs of training examples)
            if self.mixup_alpha > 0:
                mixed_images, targets_a, targets_b, lam = self._mixup_data(
                    images, labels
                )
                logits = self.model(mixed_images)
                loss = lam * self.criterion(logits, targets_a) + (
                    1 - lam
                ) * self.criterion(logits, targets_b)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )

            self.optimizer.step()

            # Step scheduler per batch (for OneCycleLR / CosineAnnealingWarmRestarts)
            if self.step_scheduler_per_batch:
                self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc



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



    def fit(
        self, num_epochs: int, resume_from: str | None = None
    ) -> Dict[str, List[float]]:
        """Train for *num_epochs* epochs with early stopping.

        Args:
            num_epochs: Maximum number of epochs to train.
            resume_from: Optional path to a checkpoint file to resume
                training from.  Restores model weights, optimizer state,
                and continues from the saved epoch.

        Returns:
            ``history`` dict with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        best_val_acc = 0.0
        start_epoch = 1
        epochs_without_improvement = 0
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_model.pth")

        # Resume from checkpoint if requested
        if resume_from is not None:
            ckpt = self.load_checkpoint(resume_from)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("val_acc", 0.0)
            logger.info(
                "Resuming training from epoch %d (best_val_acc=%.4f)",
                start_epoch, best_val_acc,
            )
            print(
                f"  ↻ Resuming from epoch {start_epoch} "
                f"(best_val_acc={best_val_acc:.4f})"
            )

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            epoch_elapsed = time.perf_counter() - epoch_start

            # LR scheduling (per-epoch schedulers like ReduceLROnPlateau)
            if not self.step_scheduler_per_batch:
                self.scheduler.step(val_loss)

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            logger.info(
                "Epoch %03d | train_loss=%.4f  train_acc=%.4f | "
                "val_loss=%.4f  val_acc=%.4f | lr=%.2e | %.1fs",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                epoch_elapsed,
            )
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} | {epoch_elapsed:.1f}s"
            )

            # Always persist the latest epoch for resume / inspection
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                last_checkpoint_path,
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
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
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

        # Training summary
        total_epochs = len(history["train_loss"])
        if total_epochs > 0:
            best_epoch = int(max(range(total_epochs), key=lambda i: history["val_acc"][i])) + start_epoch
        else:
            best_epoch = start_epoch
        logger.info(
            "Training complete — %d epochs (started at %d), best val_acc=%.4f at epoch %d",
            total_epochs, start_epoch, best_val_acc, best_epoch,
        )
        resumed_str = f" (resumed from epoch {start_epoch})" if start_epoch > 1 else ""
        print(f"\n{'='*55}")
        print(f"  Training Summary{resumed_str}")
        print(f"{'='*55}")
        print(f"  Total epochs trained:  {total_epochs}")
        print(f"  Epoch range:           {start_epoch} → {start_epoch + total_epochs - 1}")
        print(f"  Best epoch:            {best_epoch}")
        print(f"  Best val accuracy:     {best_val_acc:.4f}")
        if total_epochs > 0:
            print(f"  Final train loss:      {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss:        {history['val_loss'][-1]:.4f}")
        print(f"  Final learning rate:   {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*55}\n")

        return history



    def load_checkpoint(self, path: str) -> dict:
        """Restore model, optimizer, and scheduler state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(
            "Loaded checkpoint from '%s' (epoch=%d, val_acc=%.4f)",
            path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_acc", float("nan")),
        )
        return checkpoint
