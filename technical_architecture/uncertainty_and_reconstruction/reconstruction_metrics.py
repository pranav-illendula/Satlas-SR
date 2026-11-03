"""
Evaluation & Quantitative Comparison Metrics for Reconstructed Satellite Imagery.

Contains utilities for computing:
- Central interior window PSNR and MSE (excluding registration border artifacts)
- Integer registration shift estimation (cPSNR-aligned grid search)
- Directional edge preservation and Sobel gradient MAE against high-resolution references
- Sharpness retention ratios compared to uniform multi-temporal ensembling
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
from .soft_reconstruction import compute_sobel_gradient


def compute_registration_shift(
    pred: np.ndarray,
    ref: np.ndarray,
    max_shift: int = 8,
    border: int = 4
) -> Tuple[int, int, float]:
    """
    Find integer translation shift (dx, dy) in [-max_shift, max_shift] minimizing MSE
    on the interior crop, matching the official cPSNR evaluation protocol.
    
    Args:
        pred: Prediction image (H, W, C) in [0, 1] or [0, 255].
        ref: Ground-truth reference image (H, W, C).
        max_shift: Maximum translation search radius in pixels.
        border: Border margin to exclude from evaluation.
        
    Returns:
        (best_dx, best_dy, min_mse)
    """
    H, W = pred.shape[:2]
    pred_crop = pred[border:H-border, border:W-border]
    cH, cW = pred_crop.shape[:2]

    best_mse = float("inf")
    best_dx, best_dy = 0, 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y_start = border + dy
            y_end = y_start + cH
            x_start = border + dx
            x_end = x_start + cW

            if y_start < 0 or y_end > H or x_start < 0 or x_end > W:
                continue

            ref_crop = ref[y_start:y_end, x_start:x_end]
            mse = float(np.mean((pred_crop - ref_crop) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, best_mse


def evaluate_reconstruction_quality(
    pred: np.ndarray,
    ref: np.ndarray,
    dx: int = 0,
    dy: int = 0,
    border: int = 4,
    crop_size: int = 120,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    Evaluate structural and perceptual quality of a reconstructed image
    against an aligned reference ground truth on the central interior crop.
    
    Args:
        pred: Reconstructed prediction (H, W, C) in [0, data_range].
        ref: Reference ground truth (H, W, C).
        dx: Registration shift in x (horizontal).
        dy: Registration shift in y (vertical).
        border: Canonical border margin (e.g. 4 pixels).
        crop_size: Dimension of interior evaluation window (e.g. 120 pixels).
        data_range: Dynamic range of input pixels (1.0 or 255.0).
        
    Returns:
        Dictionary containing PSNR, MSE, MAE, edge sharpness (mean gradient), and edge MAE.
    """
    # Extract aligned crops
    p_crop = pred[border:border+crop_size, border:border+crop_size]
    r_crop = ref[border+dy:border+dy+crop_size, border+dx:border+dx+crop_size]

    diff = p_crop.astype(np.float64) - r_crop.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    
    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(10.0 * np.log10((data_range ** 2) / mse))

    # Gradient edge metrics
    p_grad = compute_sobel_gradient(p_crop)
    r_grad = compute_sobel_gradient(r_crop)

    grad_mean = float(np.mean(p_grad))
    grad_mae = float(np.mean(np.abs(p_grad - r_grad)))

    return {
        "psnr_db": psnr,
        "mse": mse,
        "mae": mae,
        "edge_sharpness": grad_mean,
        "edge_mae": grad_mae,
    }

