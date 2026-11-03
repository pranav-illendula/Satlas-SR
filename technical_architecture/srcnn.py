"""
Super-Resolution Convolutional Neural Network (SRCNN) baseline model.
Adapted from Dong et al. (2014) and WorldStrat (Cornebise et al., 2022).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class SRCNN(nn.Module):
    """
    Multi-channel SRCNN for satellite image super-resolution.
    Treats multiple temporal revisits as concatenated channel dimensions.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        hidden_channels: int = 64,
        kernel_size: int = 9,
        sr_kernel_size: int = 5,
        zoom_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.zoom_factor = zoom_factor
        pad_1 = kernel_size // 2
        pad_2 = 1
        pad_3 = sr_kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=pad_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, out_channels, sr_kernel_size, padding=pad_3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic upsampling to target spatial dimensions
        x_upsampled = F.interpolate(x, scale_factor=self.zoom_factor, mode="bicubic", align_corners=False)
        return self.net(x_upsampled)

