"""
U-Net Discriminator with Spectral Normalization (SSR_UNetDiscriminatorSN).
Adapted from Real-ESRGAN (Wang et al., 2021) and Satlas-SR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class SSR_UNetDiscriminatorSN(nn.Module):
    """
    U-Net style Discriminator with Spectral Normalization (SN).
    Unlike standard scalar discriminators that output a single classification probability,
    the U-Net discriminator outputs dense, per-pixel discrimination maps.
    This provides rich spatial gradient feedback to the generator, penalizing structural
    hallucinations while retaining fine satellite textures.

    Args:
        num_in_ch (int): Channels of input imagery (default: 3 for RGB).
        num_feat (int): Base feature channels (default: 64).
        skip_connection (bool): Whether to include U-Net encoder-decoder skip connections (default: True).
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        # Input convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # Downsampling encoder (stride 2)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsampling decoder with bilateral interpolation
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Final pixel-level classifier head
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # Decoder with U-Net skip additions
        x3_up = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3_up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4_up = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4_up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5_up = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5_up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        # Pixel classification map
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out

