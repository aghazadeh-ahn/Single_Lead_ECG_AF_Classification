"""
src/models/baseline_cnn.py
Baseline 1D Convolutional Neural Network for ECG Classification.
Input shape expected: (Batch_Size, 1, Signal_Length) -> e.g., (32, 1, 9000)
Output shape: (Batch_Size, Num_Classes)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import NUM_CLASSES


class ConvBlock1D(nn.Module):
    """
    Standard Conv1D Block: Conv1D -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # 'same' padding logic: padding = kernel_size // 2
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Baseline1DCNN(nn.Module):
    """
    4-Stage 1D CNN Architecture for Single-Lead ECG Classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.25,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extraction Backbone
        self.features = nn.Sequential(
            # Stage 1: Initial feature extraction with wide receptive field
            ConvBlock1D(in_channels=in_channels, out_channels=32, kernel_size=11, pool_size=4, dropout=dropout),
            # Stage 2
            ConvBlock1D(in_channels=32, out_channels=64, kernel_size=7, pool_size=4, dropout=dropout),
            # Stage 3
            ConvBlock1D(in_channels=64, out_channels=128, kernel_size=5, pool_size=2, dropout=dropout),
            # Stage 4
            ConvBlock1D(in_channels=128, out_channels=256, kernel_size=5, pool_size=2, dropout=dropout),
        )

        # Global Pooling & Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shape: if (B, L) -> unsqueeze to (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        features = self.features(x)
        pooled = self.global_pool(features).squeeze(-1)  # Shape: (Batch, 256)
        logits = self.classifier(pooled)                # Shape: (Batch, Num_Classes)
        return logits


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline1DCNN(in_channels=1, num_classes=4).to(device)
    
    # Simulate a batch of 4 ECG signals (30 seconds @ 300Hz = 9000 samples)
    dummy_input = torch.randn(4, 1, 9000).to(device)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(" Baseline1DCNN sanity check passed!")
    print(f"   Input shape : {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total trainable parameters: {total_params:,}")
