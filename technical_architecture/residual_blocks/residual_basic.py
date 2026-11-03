"""
Standard Residual Blocks with and without Batch Normalization.
ResidualBlockNoBN is essential for super-resolution because BatchNorm alters batch contrast statistics
and creates noticeable halo/checkerboard artifacts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .residual_dense_block import default_init_weights


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with optional Batch Normalization or Spectral Normalization.
    """

    def __init__(
        self,
        num_feat: int = 64,
        res_scale: float = 1.0,
        use_bn: bool = False,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.res_scale = res_scale
        self.use_bn = use_bn

        norm_fn = nn.utils.spectral_norm if spectral_norm else lambda m: m
        self.conv1 = norm_fn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=not use_bn))
        self.conv2 = norm_fn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=not use_bn))

        self.bn1 = nn.BatchNorm2d(num_feat) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(num_feat) if use_bn else nn.Identity()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        default_init_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return identity + out * self.res_scale


class ResidualBlockNoBN(nn.Module):
    """
    Residual Block without Batch Normalization (RCAN / EDSR / ESRGAN style).
    Removes BN layers to preserve absolute pixel intensity and prevent contrast distortion.
    """

    def __init__(self, num_feat: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        default_init_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out * self.res_scale

