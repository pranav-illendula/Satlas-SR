"""
Technical Architecture Suite for Satlas Super-Resolution.
Provides modular subpackages for Residual Blocks, GAN Architectures, and Baseline Networks.
"""
from __future__ import annotations

from .residual_blocks import (
    ResidualDenseBlock,
    RRDB,
    ResidualBlock,
    ResidualBlockNoBN,
    DenseBlock,
    FusionBlock,
    RecursiveFusion,
    PixelShuffleBlock,
    ConvTransposeBlock,
    Upsample,
)
from .gan_architecture import (
    SSR_RRDBNet,
    SSR_UNetDiscriminatorSN,
    OSMObjDiscriminator,
    SSRESRGANModel,
    RelativisticAverageGANLoss,
)
from .srcnn import SRCNN
from .highresnet import HighResNet

__all__ = [
    # Residual blocks
    "ResidualDenseBlock",
    "RRDB",
    "ResidualBlock",
    "ResidualBlockNoBN",
    "DenseBlock",
    "FusionBlock",
    "RecursiveFusion",
    "PixelShuffleBlock",
    "ConvTransposeBlock",
    "Upsample",
    # GAN Architecture
    "SSR_RRDBNet",
    "SSR_UNetDiscriminatorSN",
    "OSMObjDiscriminator",
    "SSRESRGANModel",
    "RelativisticAverageGANLoss",
    # Baselines
    "SRCNN",
    "HighResNet",
]

