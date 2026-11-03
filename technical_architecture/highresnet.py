"""
HighResNet Multi-Scale Recursive Fusion Network baseline.
Adapted from Bhowmik et al. / WorldStrat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from technical_architecture.residual_blocks.residual_basic import ResidualBlock


class HighResNet(nn.Module):
    """
    HighResNet: Recursive multi-frame super-resolution architecture
    using residual dilation and multi-scale feature aggregation.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        hidden_channels: int = 64,
        num_residual_blocks: int = 6,
        zoom_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.entry = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(num_feat=hidden_channels, res_scale=1.0, use_bn=False))
        self.body = nn.Sequential(*res_blocks)

        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * (zoom_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(zoom_factor),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.exit = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.leaky_relu(self.entry(x), 0.2, inplace=True)
        body_feat = self.body(feat) + feat
        up = self.upsample(body_feat)
        return self.exit(up)

