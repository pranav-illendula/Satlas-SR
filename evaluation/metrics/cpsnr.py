"""
Cyclic Peak Signal-to-Noise Ratio (cPSNR) metric implementation.
Adapted from the ESA PROBA-V Super-Resolution Challenge scoring protocol:
https://kelvins.esa.int/proba-v-super-resolution/scoring/
"""
from __future__ import annotations

import numpy as np


def reorder_image(img: np.ndarray, input_order: str = "HWC") -> np.ndarray:
    """Ensure image is formatted as HWC."""
    if input_order == "CHW":
        return np.ascontiguousarray(img.transpose(1, 2, 0))
    return np.ascontiguousarray(img)


def calculate_cpsnr(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    input_order: str = "HWC",
    max_offset: int = 8,
) -> float:
    """
    Computes translation- and brightness-bias-corrected PSNR (cPSNR).
    Finds the optimal spatial translation within [-max_offset, +max_offset]
    and channel brightness offset that minimizes mean squared error.

    Args:
        img (np.ndarray): Super-resolved prediction array (H, W, C) in [0, 255].
        img2 (np.ndarray): High-resolution reference array (H, W, C) in [0, 255].
        crop_border (int): Border pixels to crop before evaluation (default: 4).
        input_order (str): 'HWC' or 'CHW'.
        max_offset (int): Maximum spatial shift grid radius in pixels (default: 8).

    Returns:
        float: Best cPSNR score in decibels (dB).
    """
    assert img.shape == img2.shape, f"Image shapes must match: {img.shape} vs {img2.shape}"
    img1 = reorder_image(img, input_order=input_order)
    img2_arr = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2_arr = img2_arr[crop_border:-crop_border, crop_border:-crop_border, ...]

    img1 = img1.astype(np.float64)
    img2_arr = img2_arr.astype(np.float64)

    height, width = img1.shape[0], img1.shape[1]
    crop_height, crop_width = height - max_offset, width - max_offset
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(f"Cropped dimensions non-positive ({crop_height}x{crop_width}) for max_offset={max_offset}")

    best_mse = None
    for row_offset in range(max_offset + 1):
        for col_offset in range(max_offset + 1):
            cur1 = img1[row_offset : row_offset + crop_height, col_offset : col_offset + crop_width].copy()
            cur2 = img2_arr[
                (max_offset - row_offset) : (max_offset - row_offset) + crop_height,
                (max_offset - col_offset) : (max_offset - col_offset) + crop_width,
            ].copy()

            # Channel-wise brightness bias compensation
            channels = cur1.shape[2] if cur1.ndim == 3 else 1
            if cur1.ndim == 3:
                for c in range(channels):
                    bias = np.mean(cur1[:, :, c] - cur2[:, :, c])
                    cur2[:, :, c] += bias
            else:
                bias = np.mean(cur1 - cur2)
                cur2 += bias

            mse = np.mean(np.square(cur1 - cur2))
            if best_mse is None or mse < best_mse:
                best_mse = mse

    if best_mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 * 255.0 / best_mse))

