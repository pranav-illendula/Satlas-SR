"""
Object-Aware Discriminator with Self-Attention (OSMObjDiscriminator).
Conditioned on OpenStreetMap spatial vector masks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class SelfAttentionBlock(nn.Module):
    """Non-local Self-Attention block (Zhang et al., SAGAN)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.theta = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.phi = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.g = spectral_norm(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1))
        self.o = spectral_norm(nn.Conv2d(in_channels // 2, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, self.in_channels // 8, h * w).permute(0, 2, 1)
        phi = F.max_pool2d(self.phi(x), [2, 2]).view(b, self.in_channels // 8, (h * w) // 4)
        attn = F.softmax(torch.bmm(theta, phi), dim=-1)

        g = F.max_pool2d(self.g(x), [2, 2]).view(b, self.in_channels // 2, (h * w) // 4)
        attn_g = torch.bmm(g, attn.permute(0, 2, 1)).view(b, self.in_channels // 2, h, w)
        out = self.o(attn_g)
        return self.gamma * out + x


class OSMObjDiscriminator(nn.Module):
    """
    Discriminator network evaluating satellite imagery conditioned on OpenStreetMap object category masks.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64):
        super().__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.attn = SelfAttentionBlock(num_feat * 4)
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False))
        self.classifier = nn.Conv2d(num_feat * 8, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        h = F.leaky_relu(self.conv1(h), 0.2, inplace=True)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=True)
        h = self.attn(h)
        h = F.leaky_relu(self.conv3(h), 0.2, inplace=True)
        h = F.leaky_relu(self.conv4(h), 0.2, inplace=True)
        return self.classifier(h)

