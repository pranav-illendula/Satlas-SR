"""
Temporal Uncertainty Estimation via Multi-Revisit Frame Subsampling.

Measures aleatoric temporal volatility (transient clouds, shadows, seasonal phenology,
moving vehicles, and water surface glitter) by evaluating predictive variance across
subsampled or perturbed revisit sequences (S0 through S6).
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Union, Optional


def generate_temporal_subsamples(
    available_indices: List[int],
    num_subsamples: int = 6,
    frames_per_sample: int = 8,
    seed: Optional[int] = 0
) -> List[List[int]]:
    """
    Generate deterministic temporal revisit index subsets for uncertainty evaluation.
    
    Args:
        available_indices: List of available revisit frame indices (e.g. range(32)).
        num_subsamples: Number of alternative temporal configurations to generate (S1..SK).
        frames_per_sample: Number of temporal frames required per model forward pass (e.g. 8).
        seed: Random seed for reproducible subsampling.
        
    Returns:
        List of index lists, each containing `frames_per_sample` frames.
    """
    if len(available_indices) < frames_per_sample:
        raise ValueError(
            f"Cannot select {frames_per_sample} frames from {len(available_indices)} available."
        )

    rng = np.random.RandomState(seed)
    subsamples = []
    
    # S0: Uniformly spaced or first N frames (canonical baseline)
    step = len(available_indices) // frames_per_sample
    s0 = [available_indices[i * step] for i in range(frames_per_sample)]
    subsamples.append(s0)

    # S1..SK: Perturbed, random, or stratified selections
    for _ in range(num_subsamples):
        # Choose with replacement or sorted random choice without replacement
        sample = sorted(rng.choice(available_indices, size=frames_per_sample, replace=False).tolist())
        subsamples.append(sample)

    return subsamples


def compute_temporal_uncertainty(
    predictions_stack: Union[List[np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel-wise temporal uncertainty from an ensemble of multi-revisit predictions.
    
    Args:
        predictions_stack: List or 4D array of shape (K, H, W, C) containing predictions
                           generated from different temporal subsets (e.g. S0 through S6).
                           Values should be normalized in [0, 1] or [0, 255].
                           
    Returns:
        u_temporal: 2D array of shape (H, W) measuring root-mean-square temporal standard deviation.
        ensemble_mean: 3D array of shape (H, W, C) representing the temporal ensemble average.
    """
    if isinstance(predictions_stack, list):
        stack = np.stack(predictions_stack, axis=0)  # (K, H, W, C)
    else:
        stack = predictions_stack

    if stack.ndim != 4:
        raise ValueError(f"Expected 4D array (K, H, W, C), got shape {stack.shape}")

    k_samples = stack.shape[0]
    ddof = 1 if k_samples > 1 else 0

    # Temporal ensemble mean
    ensemble_mean = np.mean(stack, axis=0)  # (H, W, C)

    # Variance across temporal revisit predictions per channel
    var_temporal_c = np.var(stack.astype(np.float64), axis=0, ddof=ddof)  # (H, W, C)
    var_temporal_c = np.maximum(0.0, var_temporal_c)

    # Root mean square across channels: U_temporal = sqrt(1/C sum_c Var_c)
    u_temporal = np.sqrt(np.mean(var_temporal_c, axis=-1)).astype(np.float32)  # (H, W)

    return u_temporal, ensemble_mean.astype(np.float32)
