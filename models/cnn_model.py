"""CNN architecture for handwritten digit recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
INPUT_HEIGHT = 43
INPUT_WIDTH = 17


class DigitCNN(nn.Module):
    """Lightweight CNN for 17×43 grayscale digit images.

    Input shape: [B, 1, 43, 17] — batch × channels × height × width
    Output: raw logits of shape [B, 10]
    """

    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Block 1: [B, 1, 43, 17] → [B, 32, 21, 8]
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: [B, 32, 21, 8] → [B, 64, 10, 4]
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: [B, 64, 10, 4] → [B, 128, 5, 2]
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Fully-connected head
        # Flatten: 128 × 5 × 2 = 1280
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=max(dropout_rate - 0.1, 0.1))

        # Apply weight initialization
        self._init_weights()

        # Print parameter count on init
        print(f"[DigitCNN] Trainable parameters: {self.count_parameters():,}")

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming init to Conv layers, Xavier to Linear, constants to BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, start_dim=1)  # → [B, 1280]

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # raw logits

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for each class."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self, only_trainable: bool = True) -> int:
        """Return the number of model parameters.

        Args:
            only_trainable: If True, count only parameters with
                ``requires_grad=True``.  Set to False to include frozen
                parameters as well.
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Transfer-learning helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all convolutional blocks so only the FC head is trained.

        Useful for fine-tuning on a small custom digit dataset where you
        want to keep the learned feature extractor intact.
        """
        for block in (self.block1, self.block2, self.block3):
            for param in block.parameters():
                param.requires_grad = False
        frozen = self.count_parameters(only_trainable=False) - self.count_parameters()
        print(f"[DigitCNN] Backbone frozen — {frozen:,} params locked")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all convolutional blocks for full end-to-end training."""
        for block in (self.block1, self.block2, self.block3):
            for param in block.parameters():
                param.requires_grad = True
        print(f"[DigitCNN] Backbone unfrozen — {self.count_parameters():,} trainable params")

    # ------------------------------------------------------------------
    # Layer introspection
    # ------------------------------------------------------------------

    def get_layer_info(self) -> List[Dict[str, object]]:
        """Return structured information about each named module.

        Returns:
            List of dicts with keys ``name``, ``type``, ``params``,
            ``trainable``, and ``shape`` (weight shape if applicable).
        """
        info: List[Dict[str, object]] = []
        for name, module in self.named_modules():
            if name == "":  # skip root
                continue
            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )
            weight_shape: Tuple[int, ...] | None = None
            if hasattr(module, "weight") and module.weight is not None:
                weight_shape = tuple(module.weight.shape)
            info.append({
                "name": name,
                "type": module.__class__.__name__,
                "params": params,
                "trainable": trainable,
                "shape": weight_shape,
            })
        return info

    def summary(self) -> str:
        """Return a Keras-style layer summary with parameter counts.

        Returns:
            Multi-line string table showing each layer's type, output shape
            (estimated), and parameter count.
        """
        lines: List[str] = []
        sep = "-" * 65
        header = f"{'Layer':<30} {'Type':<20} {'Params':>12}"
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        total = 0
        for name, param in self.named_parameters():
            count = param.numel()
            total += count
            layer_type = "weight" if "weight" in name else "bias"
            lines.append(f"{name:<30} {layer_type:<20} {count:>12,}")

        lines.append(sep)
        lines.append(f"{'Total trainable params':<30} {'':20} {total:>12,}")
        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DigitCNN(dropout_rate={self.dropout_rate}, "
            f"params={self.count_parameters():,})"
        )
