"""
ESRGAN Generator Network (SSR_RRDBNet) for Multi-Frame Satellite Super-Resolution.
Adapted from Satlas-SR (Allen Institute for AI) and ESRGAN (Wang et al., 2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from technical_architecture.residual_blocks.rrdb import RRDB


def make_layer(basic_block, num_basic_block, **kwargs):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Inverse operation of pixel shuffle to downscale spatial dimensions and enlarge channel depth.
    """
    b, c, h, w = x.shape
    out_channel = c * (scale ** 2)
    out_h = h // scale
    out_w = w // scale
    x_view = x.contiguous().view(b, c, out_h, scale, out_w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)


class SSR_RRDBNet(nn.Module):
    """
    Enhanced Super-Resolution Generator Network with 23 RRDB trunk blocks.
    Customized for multi-frame multi-spectral satellite imagery (Sentinel-2 to NAIP).

    Args:
        num_in_ch (int): Total input channels across stacked temporal frames (e.g. 24 for 8 RGB revisits).
        num_out_ch (int): Output channels (default: 3 for high-res RGB NAIP).
        scale (int): Upsampling scale factor (default: 4).
        num_feat (int): Intermediate feature channels (default: 64).
        num_block (int): Number of RRDB trunk blocks (default: 23).
        num_grow_ch (int): Channel growth within each dense block (default: 32).
    """

    def __init__(
        self,
        num_in_ch: int = 24,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch

        # Handle scale 2 or scale 1 via pixel unshuffle if needed
        in_channels = num_in_ch
        if scale == 2:
            in_channels = num_in_ch * 4
        elif scale == 1:
            in_channels = num_in_ch * 16

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        # Deep RRDB trunk backbone
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 4x Upsampling Head (2x nearest interpolation + conv stages)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        if self.scale in [8, 16]:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.scale == 16:
                self.conv_up4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # High-resolution output reconstruction
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # Global residual skip

        # 4x upsampling stages
        if self.scale in [2, 4, 8, 16]:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))

        if self.scale == 3:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=3, mode="nearest")))

        if self.scale in [4, 8, 16]:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.scale in [8, 16]:
                feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
                if self.scale == 16:
                    feat = self.lrelu(self.conv_up4(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

