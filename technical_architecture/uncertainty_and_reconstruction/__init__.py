"""
Uncertainty Calibration & Guided Reconstruction Package.

Contains modular components for:
- Spatial uncertainty estimation via D4 dihedral Test-Time Augmentation (TTA)
- Temporal uncertainty estimation via multi-revisit subsampling and perturbation
- Calibrated hybrid uncertainty map synthesis via standardized z-score fusion
- Adaptive uncertainty-guided soft-blending reconstruction under budget constraints
- Quantitative quality, edge sharpness, and registration metrics
"""
from .spatial_uncertainty import (
    apply_d4_forward_numpy,
    apply_d4_inverse_numpy,
    compute_spatial_uncertainty_from_tta,
    run_d4_tta_inference,
)
from .temporal_uncertainty import (
    generate_temporal_subsamples,
    compute_temporal_uncertainty,
)
from .hybrid_uncertainty_map import (
    compute_hybrid_uncertainty,
    HybridUncertaintyCalibrator,
    ALPHA_DEFAULT,
    MU_SPATIAL_DEFAULT,
    STD_SPATIAL_DEFAULT,
    MU_TEMPORAL_DEFAULT,
    STD_TEMPORAL_DEFAULT,
)
from .soft_reconstruction import (
    compute_sobel_gradient,
    stable_sigmoid,
    solve_tau_for_budget,
    apply_soft_blending,
    apply_hard_thresholding,
    compute_boundary_step_discontinuity,
)
from .reconstruction_metrics import (
    compute_registration_shift,
    evaluate_reconstruction_quality,
)

__all__ = [
    # Spatial
    "apply_d4_forward_numpy",
    "apply_d4_inverse_numpy",
    "compute_spatial_uncertainty_from_tta",
    "run_d4_tta_inference",
    # Temporal
    "generate_temporal_subsamples",
    "compute_temporal_uncertainty",
    # Hybrid Map
    "compute_hybrid_uncertainty",
    "HybridUncertaintyCalibrator",
    "ALPHA_DEFAULT",
    "MU_SPATIAL_DEFAULT",
    "STD_SPATIAL_DEFAULT",
    "MU_TEMPORAL_DEFAULT",
    "STD_TEMPORAL_DEFAULT",
    # Soft Reconstruction
    "compute_sobel_gradient",
    "stable_sigmoid",
    "solve_tau_for_budget",
    "apply_soft_blending",
    "apply_hard_thresholding",
    "compute_boundary_step_discontinuity",
    # Quality Metrics
    "compute_registration_shift",
    "evaluate_reconstruction_quality",
]

