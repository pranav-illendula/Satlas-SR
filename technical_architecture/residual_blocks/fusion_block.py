"""
Fusion Block modules for merging multi-temporal Sentinel-2 revisits and multi-band spectral channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """
    Spatio-temporal fusion block for merging temporal Sentinel-2 revisits.
    Combines pointwise 1x1 convolutions with non-linear mapping.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RecursiveFusion(nn.Module):
    """
    Recursive feature aggregator across temporal image sequences.
    Iteratively fuses pairs of consecutive temporal representations.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, frame_list: list[torch.Tensor]) -> torch.Tensor:
        current = frame_list
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    fused = self.fusion(torch.cat([current[i], current[i + 1]], dim=1))
                    next_level.append(fused)
                else:
                    next_level.append(current[i])
            current = next_level
        return current[0]

