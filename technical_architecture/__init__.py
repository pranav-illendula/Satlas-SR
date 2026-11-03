"""
Technical Architecture Suite for Satlas Super-Resolution.
Provides modular subpackages for Residual Blocks, GAN Architectures, Baseline Networks,
and Uncertainty Calibration & Guided Reconstruction.
"""
from __future__ import annotations

# Uncertainty and Reconstruction (pure NumPy / SciPy compatible)
from .uncertainty_and_reconstruction import (
    apply_d4_forward_numpy,
    apply_d4_inverse_numpy,
    compute_spatial_uncertainty_from_tta,
    run_d4_tta_inference,
    generate_temporal_subsamples,
    compute_temporal_uncertainty,
    compute_hybrid_uncertainty,
    HybridUncertaintyCalibrator,
    compute_sobel_gradient,
    solve_tau_for_budget,
    apply_soft_blending,
    apply_hard_thresholding,
    compute_boundary_step_discontinuity,
    compute_registration_shift,
    evaluate_reconstruction_quality,
)

# PyTorch dependent modules (guarded for environments without PyTorch installed)
try:
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
except ImportError:
    ResidualDenseBlock = None
    RRDB = None
    ResidualBlock = None
    ResidualBlockNoBN = None
    DenseBlock = None
    FusionBlock = None
    RecursiveFusion = None
    PixelShuffleBlock = None
    ConvTransposeBlock = None
    Upsample = None
    SSR_RRDBNet = None
    SSR_UNetDiscriminatorSN = None
    OSMObjDiscriminator = None
    SSRESRGANModel = None
    RelativisticAverageGANLoss = None
    SRCNN = None
    HighResNet = None

__all__ = [
    # Uncertainty & Reconstruction
    "apply_d4_forward_numpy",
    "apply_d4_inverse_numpy",
    "compute_spatial_uncertainty_from_tta",
    "run_d4_tta_inference",
    "generate_temporal_subsamples",
    "compute_temporal_uncertainty",
    "compute_hybrid_uncertainty",
    "HybridUncertaintyCalibrator",
    "compute_sobel_gradient",
    "solve_tau_for_budget",
    "apply_soft_blending",
    "apply_hard_thresholding",
    "compute_boundary_step_discontinuity",
    "compute_registration_shift",
    "evaluate_reconstruction_quality",
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
