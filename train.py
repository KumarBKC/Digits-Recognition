"""Main training entry point."""

from __future__ import annotations

import argparse
import os
import random
import sys

# Allow running from the digit_recognition directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import DigitCNN
from training.dataset_loader import create_dataloaders
from training.trainer import Trainer
from utils import visualizer
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DigitCNN")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/dataset",
        help="Path to dataset directory containing train/ and val/ sub-folders.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Seed everything for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # 2. Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)
    print(f"Device: {device}")

    # 3. Create dataloaders + class weights
    logger.info("Loading dataset from '%s' ...", args.data_root)
    train_loader, val_loader, class_weights = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=2,
    )
    logger.info(
        "Dataset: %d train batches, %d val batches",
        len(train_loader),
        len(val_loader),
    )

    # 4. Instantiate model
    model = DigitCNN(dropout_rate=0.4).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # 5. Loss: weighted CrossEntropy (handles class imbalance)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 6. Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 7. Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 8. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )
    history = trainer.fit(args.epochs)

    # 9. Plot training curves + confusion matrix
    visualizer.plot_history(history)
    visualizer.plot_confusion_matrix(model, val_loader, device)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
