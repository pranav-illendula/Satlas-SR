"""
Spatial Uncertainty Estimation via D4 Test-Time Augmentation (TTA).

Measures epistemic uncertainty and lack of rotational equivariance in deep
convolutional super-resolution models by perturbing input imagery through the
dihedral symmetry group D4 (8 transformations: 4 rotations x 2 reflections).
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Union, Optional


def apply_d4_forward_numpy(img: np.ndarray, k: int, f: int) -> np.ndarray:
    """
    Apply D4 forward transformation to a NumPy array (H, W, C) or (H, W).
    
    Args:
        img: Input array of shape (H, W, C) or (H, W).
        k: Number of 90-degree counter-clockwise rotations in {0, 1, 2, 3}.
        f: Reflection flag in {0, 1} (1 = horizontal flip across width).
        
    Returns:
        Transformed array.
    """
    out = img.copy()
    if f == 1:
        out = np.flip(out, axis=1)  # horizontal flip across width
    if k != 0:
        out = np.rot90(out, k=k, axes=(0, 1))
    return out


def apply_d4_inverse_numpy(img: np.ndarray, k: int, f: int) -> np.ndarray:
    """
    Apply inverse D4 transformation to restore original spatial orientation.
    
    Args:
        img: Transformed array of shape (H, W, C) or (H, W).
        k: Rotation parameter used during forward pass in {0, 1, 2, 3}.
        f: Reflection parameter used during forward pass in {0, 1}.
        
    Returns:
        Restored array in original coordinate frame.
    """
    out = img.copy()
    if k != 0:
        inv_k = (4 - k) % 4
        out = np.rot90(out, k=inv_k, axes=(0, 1))
    if f == 1:
        out = np.flip(out, axis=1)
    return out


def compute_spatial_uncertainty_from_tta(
    tta_predictions: Union[List[np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial predictive uncertainty from an ensemble of 8 D4 inverted predictions.
    
    Args:
        tta_predictions: Stack or list of 8 inverted prediction arrays, each (H, W, C) in [0, 1] or [0, 255].
        
    Returns:
        u_spatial: 2D uncertainty map (H, W) measuring root-mean-square variance across channels.
        mean_pred: Mean prediction across all 8 orientations (H, W, C).
    """
    if isinstance(tta_predictions, list):
        stack = np.stack(tta_predictions, axis=0)  # (N, H, W, C)
    else:
        stack = tta_predictions

    if stack.ndim != 4:
        raise ValueError(f"Expected 4D array of predictions (N, H, W, C), got shape {stack.shape}")

    # Compute mean and variance across TTA variations (axis 0)
    mean_pred = np.mean(stack, axis=0)  # (H, W, C)
    var_per_channel = np.var(stack.astype(np.float64), axis=0, ddof=1 if stack.shape[0] > 1 else 0)  # (H, W, C)
    var_per_channel = np.maximum(0.0, var_per_channel)
    
    # Root mean square across color channels: U_spatial = sqrt(1/C sum_c Var_c)
    u_spatial = np.sqrt(np.mean(var_per_channel, axis=-1)).astype(np.float32)  # (H, W)
    
    return u_spatial, mean_pred.astype(np.float32)


def run_d4_tta_inference(
    model,
    lr_stack,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Execute full D4 Test-Time Augmentation on a PyTorch model and multi-frame input.
    
    Args:
        model: PyTorch super-resolution model (e.g. SSR_RRDBNet).
        lr_stack: Low-resolution multi-frame tensor (1, C_in, H, W).
        device: Target execution device ('cuda' or 'cpu').
        
    Returns:
        u_spatial: (H_out, W_out) spatial uncertainty map.
        mean_pred: (H_out, W_out, 3) ensemble mean prediction in [0, 1].
        preds_list: List of 8 individual inverted predictions.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for model-in-the-loop run_d4_tta_inference.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = model.to(dev).eval()
    lr_stack = lr_stack.to(dev)

    inverted_preds = []
    d4_transforms = [(k, f) for f in [0, 1] for k in [0, 1, 2, 3]]

    with torch.no_grad():
        for k, f in d4_transforms:
            # 1. Apply forward D4 spatial augmentation to LR multi-frame input
            x_aug = lr_stack
            if f == 1:
                x_aug = torch.flip(x_aug, dims=[-1])
            if k != 0:
                x_aug = torch.rot90(x_aug, k=k, dims=[-2, -1])

            # 2. Forward pass through generator
            pred_aug = model(x_aug)

            # 3. Apply inverse D4 transformation to align back with canonical grid
            if k != 0:
                inv_k = (4 - k) % 4
                pred_aug = torch.rot90(pred_aug, k=inv_k, dims=[-2, -1])
            if f == 1:
                pred_aug = torch.flip(pred_aug, dims=[-1])

            # Convert to (H, W, C) float32 numpy array
            arr = pred_aug.squeeze(0).permute(1, 2, 0).cpu().numpy()
            inverted_preds.append(np.clip(arr, 0.0, 1.0))

    u_spatial, mean_pred = compute_spatial_uncertainty_from_tta(inverted_preds)
    return u_spatial, mean_pred, inverted_preds
