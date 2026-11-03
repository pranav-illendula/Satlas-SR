"""
Residual in Residual Dense Block (RRDB) for deep ESRGAN feature extraction.
Adapted from ESRGAN (Wang et al., 2018) and Satlas-SR.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .residual_dense_block import ResidualDenseBlock


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    Combines three densely connected ResidualDenseBlock modules inside an outer residual skip connection.
    This multi-level residual network design enables very deep trunk backbones (e.g. 23 blocks)
    without vanishing gradients or training instability.

    Mathematical formulation:
      out = x + 0.2 * RDB3(RDB2(RDB1(x)))

    Args:
        num_feat (int): Intermediate feature channels (default: 64).
        num_grow_ch (int): Growth channels within each RDB (default: 32).
        res_scale (float): Outer residual scale multiplier (default: 0.2).
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale=res_scale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale=res_scale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, res_scale=res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x

