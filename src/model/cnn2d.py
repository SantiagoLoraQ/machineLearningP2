"""Arquitectura CNN 2D en PyTorch para clasificación multiclase de aves."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class BirdCNN2D(nn.Module):
    """Red Neuronal Convolucional 2D para espectrogramas de Mel (1, 64, 173)."""

    def __init__(self, num_classes: int = 39, dropout: float = 0.35) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.features = nn.Sequential(
            # Bloque 1 (stem stride-2): (1, 64, 173) -> (24, 16, 43)
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10),
            # Bloque 2: (24, 16, 43) -> (48, 8, 21)
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),
            # Bloque 3: (48, 8, 21) -> (96, 4, 10)
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),
            # Bloque 4: (96, 4, 10) -> (192, 1, 1)
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        head_dropout_2 = max(0.10, round(dropout - 0.10, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(192, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout_2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga un lote de espectrogramas (B, 1, F, T) y devuelve logits (B, C)."""
        feats = self.features(x)
        return self.classifier(feats)

    def count_parameters(self) -> int:
        """Cuenta el total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> dict[str, Any]:
        """Devuelve un resumen estructurado de la arquitectura para artefactos/UI."""
        return {
            "model_name": "BirdCNN2D",
            "input_shape": [1, 64, 173],
            "conv_channels": [24, 48, 96, 192],
            "classifier_hidden": 128,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "trainable_parameters": self.count_parameters(),
        }
