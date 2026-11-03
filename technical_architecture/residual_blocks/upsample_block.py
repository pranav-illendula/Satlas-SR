"""
Spatial Upsampling Blocks for Super-Resolution Generators.
Includes Sub-Pixel Convolution (PixelShuffle), ConvTranspose, and Nearest-Neighbor Interpolation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class PixelShuffleBlock(nn.Module):
    """
    Sub-pixel convolution upsampler (Shi et al., 2016).
    Applies Conv2d with num_feat * (upscale_factor**2) channels followed by nn.PixelShuffle.
    """

    def __init__(self, num_feat: int, upscale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(num_feat, num_feat * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """
    Transposed convolution upsampler (Deconvolution).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=stride * 2, stride=stride, padding=stride // 2
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.deconv(x))


class Upsample(nn.Sequential):
    """
    Nearest-neighbor or bilinear interpolation followed by a convolution layer.
    Preferred in ESRGAN over transposed convolutions to avoid checkerboard artifacts.
    """

    def __init__(self, scale: int, num_feat: int, mode: str = "nearest"):
        m = []
        if (scale & (scale - 1)) == 0:  # Scale is power of 2
            for _ in range(int(torch.log2(torch.tensor(float(scale))))):
                m.append(nn.Upsample(scale_factor=2, mode=mode))
                m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                m.append(nn.LeakyReLU(0.2, inplace=True))
        elif scale == 3:
            m.append(nn.Upsample(scale_factor=3, mode=mode))
            m.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            m.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(f"Unsupported upsampling scale: {scale}")
        super().__init__(*m)

