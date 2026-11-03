"""
Dense Block implementations for multi-scale feature reuse.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .residual_dense_block import default_init_weights


class DenseBlock(nn.Module):
    """
    Standard Dense Block (Huang et al., DenseNet).
    Each layer concatenates all preceding feature maps.
    """

    def __init__(self, num_layers: int = 4, in_channels: int = 64, growth_rate: int = 32):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.layers.append(layer)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

