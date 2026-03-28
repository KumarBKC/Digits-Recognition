"""CNN architecture for handwritten digit recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc3 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=max(dropout_rate - 0.1, 0.1))

        # Print parameter count on init
        print(f"[DigitCNN] Trainable parameters: {self.count_parameters():,}")

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

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
