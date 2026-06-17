"""Main training entry point."""

from __future__ import annotations

import argparse
import os
import random
import sys

# Allow running from the digit_recognition directory
sys.path.insert(0, os.path.dirname(__file__))

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import DigitCNN
from training.dataset_loader import DigitDataset, create_dataloaders, print_dataset_summary
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
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.2,
        help="Mixup interpolation strength. 0 = disabled. (default: 0.2)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CrossEntropyLoss. (default: 0.1)",
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    print_dataset_summary(
        cast(DigitDataset, train_loader.dataset), 
        cast(DigitDataset, val_loader.dataset), 
        class_weights
    )
    logger.info(
        "Dataset: %d train batches, %d val batches",
        len(train_loader),
        len(val_loader),
    )

    # 4. Instantiate model
    model = DigitCNN(dropout_rate=0.4).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # 5. Loss: weighted CrossEntropy with label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    # 6. Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 7. Scheduler: OneCycleLR (step per batch for smooth cosine annealing)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 3,   # peak LR = 3× base
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,        # 30% warmup
        anneal_strategy="cos",
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
        mixup_alpha=args.mixup_alpha,
        step_scheduler_per_batch=True,
    )
    history = trainer.fit(args.epochs)

    # 9. Plot training curves + confusion matrix
    visualizer.plot_history(history)
    visualizer.plot_confusion_matrix(model, val_loader, device)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
