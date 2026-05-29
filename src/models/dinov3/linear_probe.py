"""
DINOv3 global linear probe (image classification head from the DINOv3 paper).

Applies a LayerNorm to the final CLS-token feature and trains a single Linear
classifier on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import trunc_normal


class DINOv3LinearProbe(nn.Module):
    """DINOv3 global linear probe (image classification).

    The DINOv3 evaluation protocol applies a LayerNorm to the final CLS-token
    feature and trains a single Linear classifier on top.

    Parameters
    ----------
    embed_dim : int
        Backbone embedding dimension (e.g. 1024 for ViT-L/16, 1536 for ViT-H/16).
    num_classes : int
        Number of output classes. Defaults to 3 (background / tackle-live /
        tackle-replay).
    """

    def __init__(self, embed_dim: int, num_classes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_token : torch.Tensor
            Shape ``(B, embed_dim)``: frozen CLS feature from the backbone.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(B, num_classes)``.
        """
        return self.classifier(self.norm(cls_token))
