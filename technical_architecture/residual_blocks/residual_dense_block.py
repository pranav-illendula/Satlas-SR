"""
Residual Dense Block (RDB) module for ESRGAN architectures.
Adapted from BasicSR / ESRGAN (Wang et al., 2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def default_init_weights(module_list, scale=1.0, bias_fill=0.0):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in")
            module.weight.data *= scale
            if module.bias is not None:
                module.bias.data.fill_(bias_fill)
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                module.bias.data.fill_(bias_fill)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    Contains 5 convolutional layers where feature maps from all prior convolutions
    are concatenated channel-wise before the subsequent convolution.
    
    Mathematical formulation:
      x1 = sigma(Conv1(x0))
      x2 = sigma(Conv2(cat([x0, x1])))
      x3 = sigma(Conv3(cat([x0, x1, x2])))
      x4 = sigma(Conv4(cat([x0, x1, x2, x3])))
      x5 = Conv5(cat([x0, x1, x2, x3, x4]))
      out = x5 * res_scale + x0

    Args:
        num_feat (int): Intermediate feature channels (default: 64).
        num_grow_ch (int): Channel growth per convolution (default: 32).
        res_scale (float): Residual scaling multiplier (default: 0.2).
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * self.res_scale + x

