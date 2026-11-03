# Technical Architecture

This directory contains the complete modular technical architecture for Satlas Super-Resolution (Satlas-SR), organized into residual feature extractors, GAN pipelines, uncertainty calibration systems, and baseline networks.

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
│
├── gan_architecture/                   # Generative Adversarial Network components
│   ├── generator.py                    # SSR_RRDBNet deep generator trunk (16.7M params)
│   ├── discriminator.py                # SSR_UNetDiscriminatorSN with Spectral Normalization
│   ├── osm_discriminator.py            # Self-attention object-aware discriminator
│   ├── esrgan_gan_model.py             # RaGAN multi-task training orchestrator
│   ├── __init__.py                     # Clean package exports
│   └── README.md                       # Relativistic GAN loss & training dynamics
│
├── uncertainty_and_reconstruction/     # Spatial-temporal uncertainty & guided repair
│   ├── spatial_uncertainty.py          # D4 dihedral group Test-Time Augmentation (TTA)
│   ├── temporal_uncertainty.py         # Multi-revisit frame subsampling & volatility maps
│   ├── hybrid_uncertainty_map.py       # Standardized z-score fusion (alpha=0.19)
│   ├── soft_reconstruction.py          # Continuous budget-constrained sigmoid soft blending
│   ├── reconstruction_metrics.py       # Alignment shifts, Sobel edge MAE & sharpness metrics
│   ├── __init__.py                     # Clean package exports
│   └── README.md                       # Calibration theory, math & empirical gains
│
├── srcnn.py                            # Super-Resolution CNN baseline
├── highresnet.py                       # Recursive Multi-Scale HighResNet baseline
├── __init__.py                         # Top-level architecture exports
└── README.md                           # Architecture overview & parameter profiles
```

---

## Architectural Comparison Matrix

| Component / Subsystem | Primary Focus | Mechanism / Backbone | Key Output / Loss |
| :--- | :--- | :--- | :--- |
| **ESRGAN Generator** | 4x Multi-frame Super-Resolution | 23 RRDB Blocks (16.7M params) | $128 \times 128 \times 3$ RGB Output |
| **U-Net Discriminator** | Adversarial Realism Feedback | Spectral Norm U-Net with Skip Connections | $128 \times 128$ Pixel Discrimination Map |
| **Spatial Uncertainty** | Epistemic / Equivariance Variance | Dihedral Group $D_4$ Invariant TTA | $U_{\text{spatial}}$ (H, W) |
| **Temporal Uncertainty** | Aleatoric / Atmospheric Volatility | Revisit Subsampling ($S_0 \dots S_6$) | $U_{\text{temporal}}$ (H, W) |
| **Hybrid Calibration** | Error-Rank Calibrated Confidence | Standardized Z-Score Fusion ($\alpha=0.19$) | $U_{\text{hybrid}}$ (+39.5% holdout gain) |
| **Soft Reconstruction** | Image Repair & Artifact Removal | Continuous Sigmoid Blending ($\tau^*, \beta=0.20$) | $\hat{y}_{\text{recon}}$ (+0.571 dB PSNR gain) |

---

## Quick Usage

### 1. Generating Model Outputs & Uncertainty Maps
```python
import numpy as np
from technical_architecture.uncertainty_and_reconstruction import (
    compute_spatial_uncertainty_from_tta,
    compute_temporal_uncertainty,
    compute_hybrid_uncertainty,
    apply_soft_blending
)

# 1. Compute spatial uncertainty from 8 inverted D4 TTA predictions
u_spatial, _ = compute_spatial_uncertainty_from_tta(tta_predictions)

# 2. Compute temporal uncertainty across revisit predictions
u_temporal, ensemble_mean = compute_temporal_uncertainty(revisit_predictions)

# 3. Fuse into calibrated hybrid uncertainty map
u_hybrid, _, _ = compute_hybrid_uncertainty(u_spatial, u_temporal, alpha=0.1900)

# 4. Perform continuous soft-blending reconstruction (20% budget)
recon_img, weight_field, tau_star = apply_soft_blending(
    single_pred=s0_prediction,
    ensemble_pred=ensemble_mean,
    uncertainty_map=u_hybrid,
    budget=0.20,
    beta=0.20
)
```
