"""
Generative Adversarial Network (GAN) architecture modules for satellite super-resolution.
"""
from __future__ import annotations

from .generator import SSR_RRDBNet, pixel_unshuffle
from .discriminator import SSR_UNetDiscriminatorSN
from .osm_discriminator import OSMObjDiscriminator, SelfAttentionBlock
from .esrgan_gan_model import SSRESRGANModel, RelativisticAverageGANLoss

__all__ = [
    "SSR_RRDBNet",
    "SSR_UNetDiscriminatorSN",
    "OSMObjDiscriminator",
    "SelfAttentionBlock",
    "SSRESRGANModel",
    "RelativisticAverageGANLoss",
    "pixel_unshuffle",
]

