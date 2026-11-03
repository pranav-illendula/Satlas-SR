#!/usr/bin/env python3
"""
Spatial registration audit tool for Satlas-SR.
Quantifies spatial translation offsets between low-resolution Sentinel-2 input,
super-resolved predictions, and high-resolution NAIP reference imagery using
2D Fourier phase correlation and sub-pixel cross-correlation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import correlate2d


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.array(rgb, dtype=np.float32) / 255.0
        # Convert RGB to Luminance
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        return gray


def compute_phase_correlation(ref: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """
    Computes 2D Fourier phase cross-correlation between ref and moving arrays.
    Returns (shift_x, shift_y, peak_val).
    """
    eps = 1e-12
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(moving)
    cross_power = (f_ref * np.conj(f_mov)) / (np.abs(f_ref * np.conj(f_mov)) + eps)
    correlation = np.real(np.fft.ifft2(cross_power))

    # Find peak
    h, w = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_val = float(correlation[peak_y, peak_x])

    shift_y = peak_y if peak_y <= h // 2 else peak_y - h
    shift_x = peak_x if peak_x <= w // 2 else peak_x - w

    return float(shift_x), float(shift_y), peak_val


def compute_normalized_cross_correlation(ref: np.ndarray, moving: np.ndarray, max_offset: int = 8) -> tuple[int, int, float]:
    """
    Search integer translations within [-max_offset, +max_offset] to maximize
    normalized cross correlation on the shared central subwindow.
    """
    h, w = ref.shape
    crop_h, crop_w = h - 2 * max_offset, w - 2 * max_offset
    ref_sub = ref[max_offset : max_offset + crop_h, max_offset : max_offset + crop_w]
    ref_norm = ref_sub - np.mean(ref_sub)
    ref_std = np.std(ref_sub) + 1e-8

    best_dx, best_dy, best_ncc = 0, 0, -1.0
    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):
            mov_sub = moving[max_offset + dy : max_offset + dy + crop_h, max_offset + dx : max_offset + dx + crop_w]
            mov_norm = mov_sub - np.mean(mov_sub)
            mov_std = np.std(mov_sub) + 1e-8
            ncc = np.mean(ref_norm * mov_norm) / (ref_std * mov_std)
            if ncc > best_ncc:
                best_ncc = float(ncc)
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy, best_ncc


def audit_pair_registration(ref_path: Path, moving_path: Path, max_offset: int = 8) -> dict:
    ref = load_grayscale_image(ref_path)
    mov = load_grayscale_image(moving_path)

    # Upscale moving if dimensions differ (e.g., 32x32 S2 upscaled 4x to match 128x128 NAIP)
    if mov.shape != ref.shape:
        mov_im = Image.fromarray((mov * 255).astype(np.uint8)).resize(
            (ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR
        )
        mov = np.array(mov_im, dtype=np.float32) / 255.0

    shift_x, shift_y, phase_peak = compute_phase_correlation(ref, mov)
    best_dx, best_dy, ncc = compute_normalized_cross_correlation(ref, mov, max_offset=max_offset)

    return {
        "ref_path": str(ref_path),
        "moving_path": str(moving_path),
        "phase_corr_shift_x": shift_x,
        "phase_corr_shift_y": shift_y,
        "phase_corr_peak": phase_peak,
        "ncc_best_shift_x": best_dx,
        "ncc_best_shift_y": best_dy,
        "ncc_value": ncc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True, help="Reference image path (e.g. NAIP rgb.png)")
    parser.add_argument("--moving", type=Path, required=True, help="Moving image path (e.g. Sentinel-2 or prediction)")
    parser.add_argument("--max-offset", type=int, default=8, help="Max offset search window (default: 8)")
    args = parser.parse_args()

    result = audit_pair_registration(args.ref, args.moving, max_offset=args.max_offset)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

