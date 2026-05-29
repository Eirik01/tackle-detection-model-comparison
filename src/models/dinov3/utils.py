"""Shared utilities for DINOv3 evaluation heads."""

from __future__ import annotations

import torch
import torch.nn as nn


def trunc_normal(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal initialization clamped at +/- 2*std (DINO/MAE default)."""
    return nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2 * std, b=2 * std)

class Mlp(nn.Module):
    """Standard transformer MLP: Linear -> GeLU -> Linear (single GeLU)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
