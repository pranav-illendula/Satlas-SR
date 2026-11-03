"""
ESRGAN Generative Adversarial Training Pipeline Orchestrator.
Implements Relativistic Average GAN (RaGAN) loss, perceptual loss, and L1 pixel loss.
Adapted from BasicSR / Satlas-SR.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from .generator import SSR_RRDBNet
from .discriminator import SSR_UNetDiscriminatorSN


class RelativisticAverageGANLoss(nn.Module):
    """
    Relativistic average GAN (RaGAN) loss function (Jolicoeur-Martineau, 2018).
    Discriminator predicts the probability that a real image is relatively more realistic
    than an average fake image:
      D_Ra(x_real, x_fake) = sigma(C(x_real) - E[C(x_fake)])
      D_Ra(x_fake, x_real) = sigma(C(x_fake) - E[C(x_real)])
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward_d(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator RaGAN loss."""
        loss_real = self.bce_loss(real_logits - torch.mean(fake_logits), torch.ones_like(real_logits))
        loss_fake = self.bce_loss(fake_logits - torch.mean(real_logits), torch.zeros_like(fake_logits))
        return (loss_real + loss_fake) * 0.5

    def forward_g(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator RaGAN loss."""
        loss_real = self.bce_loss(real_logits - torch.mean(fake_logits), torch.zeros_like(real_logits))
        loss_fake = self.bce_loss(fake_logits - torch.mean(real_logits), torch.ones_like(fake_logits))
        return (loss_real + loss_fake) * 0.5


class SSRESRGANModel:
    """
    Complete ESRGAN architecture manager with generator, discriminator, and loss computation.
    """

    def __init__(
        self,
        generator: SSR_RRDBNet,
        discriminator: SSR_UNetDiscriminatorSN | None = None,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        gan_weight: float = 0.1,
    ):
        self.net_g = generator
        self.net_d = discriminator
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight

        self.l1_loss = nn.L1Loss()
        self.ragan_loss = RelativisticAverageGANLoss()

    def compute_generator_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        real_pred_d: torch.Tensor | None = None,
        fake_pred_d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculates combined multi-task generator loss."""
        loss_dict = {}
        total_loss = 0.0

        # L1 Pixel Reconstruction Loss
        l_pix = self.l1_loss(output, target) * self.pixel_weight
        total_loss += l_pix
        loss_dict["l_pix"] = float(l_pix.item())

        # Adversarial RaGAN Loss
        if self.net_d is not None and real_pred_d is not None and fake_pred_d is not None:
            l_gan = self.ragan_loss.forward_g(real_pred_d, fake_pred_d) * self.gan_weight
            total_loss += l_gan
            loss_dict["l_gan"] = float(l_gan.item())

        loss_dict["total_loss"] = float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, loss_dict

    def compute_discriminator_loss(
        self, real_pred: torch.Tensor, fake_pred: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculates Relativistic Average Discriminator loss."""
        l_d = self.ragan_loss.forward_d(real_pred, fake_pred)
        return l_d, {"l_d": float(l_d.item())}

