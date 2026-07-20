"""CNN architecture for handwritten digit recognition.

Optimized with:
  - Squeeze-and-Excitation (SE) channel attention
  - Residual connections for better gradient flow
  - Global Average Pooling replacing heavy FC layers (~90% fewer params)
"""

from __future__ import annotations



import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
INPUT_HEIGHT = 43
INPUT_WIDTH = 17


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights via global average pooling
    followed by a bottleneck FC network, then re-scales feature maps.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + skip connection.

    Maintains spatial dimensions; helps gradient flow in deeper networks.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class DigitCNN(nn.Module):
    """Optimized CNN for 17×43 grayscale digit images.

    Architecture improvements over the baseline:
      - Squeeze-and-Excitation (SE) attention after each conv block
      - Residual connections for better gradient flow
      - Global Average Pooling (GAP) replacing heavy FC layers
      - Bias-free Conv2d layers before BatchNorm

    Input shape: [B, 1, 43, 17] — batch × channels × height × width
    Output: raw logits of shape [B, 10]
    """

    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Block 1: [B, 1, 43, 17] → [B, 32, 21, 8]
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(2, 2),
        )
        self.res1 = ResidualBlock(32)

        # Block 2: [B, 32, 21, 8] → [B, 64, 10, 4]
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2),
        )
        self.res2 = ResidualBlock(64)

        # Block 3: [B, 64, 10, 4] → [B, 128, 5, 2]
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling replaces heavy FC layers
        # 128 × 5 × 2  →  128 × 1 × 1   (reduces params by ~90%)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, NUM_CLASSES)

        # Apply weight initialization
        self._init_weights()

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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.block1(x)
        x = self.res1(x)
        x = self.block2(x)
        x = self.res2(x)
        x = self.block3(x)

        x = self.gap(x)                      # → [B, 128, 1, 1]
        x = torch.flatten(x, start_dim=1)    # → [B, 128]
        x = self.dropout(x)
        x = self.classifier(x)               # raw logits

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

    @property
    def device(self) -> torch.device:
        """Get the device the model parameters are currently on."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

