# Technical Architecture

This directory contains the complete modular technical architecture for Satlas Super-Resolution (Satlas-SR).

## Directory Structure

```
technical_architecture/
├── residual_blocks/                    # Residual, dense, and temporal fusion blocks
│   ├── residual_dense_block.py         # 5-layer densely connected RDB with residual scaling
│   ├── rrdb.py                         # Hierarchical Residual-in-Residual Dense Block
│   ├── residual_basic.py               # Standard ResNet & No-BN residual blocks
│   ├── dense_block.py                  # Standard DenseNet blocks
│   ├── fusion_block.py                 # Spatio-temporal frame fusion modules
│   ├── upsample_block.py               # Sub-pixel convolution & interpolation upsamplers
│   ├── __init__.py                     # Clean package exports
│   └── README.md                       # Mathematical formulation and design rationale
├── gan_architecture/                   # Generative Adversarial Network components
│   ├── generator.py                    # SSR_RRDBNet deep generator trunk (16.7M params)
│   ├── discriminator.py                # SSR_UNetDiscriminatorSN with Spectral Normalization
│   ├── osm_discriminator.py            # Self-attention object-aware discriminator
│   ├── esrgan_gan_model.py             # RaGAN multi-task training orchestrator
│   ├── __init__.py                     # Clean package exports
│   └── README.md                       # Relativistic GAN loss & training dynamics
├── srcnn.py                            # Super-Resolution CNN baseline
├── highresnet.py                       # Recursive Multi-Scale HighResNet baseline
├── __init__.py                         # Top-level architecture exports
└── README.md                           # Architecture overview & parameter profiles
```

---

## Architectural Comparison Matrix

| Model | Primary Backbone | Upsampling Mechanism | Discriminator Type | Loss Functions |
| :--- | :--- | :--- | :--- | :--- |
| **ESRGAN (Satlas-SR)** | 23 RRDB Blocks (Trunk) | $4\times$ Nearest Interpolation + Conv | U-Net with Spectral Normalization | L1 Pixel + VGG Perceptual + Relativistic GAN |
| **HighResNet** | Recursive Residual Blocks | PixelShuffle Sub-pixel Conv | N/A (L2 / PSNR only) | L2 / L1 Pixel Reconstruction |
| **SRCNN** | 3 Conv Layers | Bicubic Pre-interpolation | N/A (L2 / PSNR only) | MSE / L2 Loss |

---

## Quick Usage

```python
import torch
from technical_architecture.residual_blocks import RRDB, ResidualDenseBlock
from technical_architecture.gan_architecture import SSR_RRDBNet, SSR_UNetDiscriminatorSN

# 1. Instantiate the 4x ESRGAN Generator (8 temporal Sentinel-2 revisit frames = 24 input channels)
generator = SSR_RRDBNet(num_in_ch=24, num_out_ch=3, scale=4, num_feat=64, num_block=23)

# 2. Instantiate U-Net Discriminator with Spectral Normalization
discriminator = SSR_UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)

# 3. Forward pass
fake_input = torch.randn(1, 24, 32, 32)
sr_output = generator(fake_input)
print("Super-resolved shape:", sr_output.shape) # [1, 3, 128, 128]

pred_d = discriminator(sr_output)
print("Discriminator map shape:", pred_d.shape) # [1, 1, 128, 128]
```

