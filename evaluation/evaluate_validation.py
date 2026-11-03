#!/usr/bin/env python3
"""
Validation evaluation and auditing script for Satlas-SR predictions.
Evaluates Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
and dynamic pixel ranges without altering prediction or reference images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

PILOT_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]


def sha256_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compute_psnr_numpy(ref: np.ndarray, pred: np.ndarray, data_range: float = 255.0) -> float:
    """Computes PSNR in pure NumPy."""
    mse = float(np.mean((ref.astype(np.float64) - pred.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim_numpy(ref: np.ndarray, pred: np.ndarray, data_range: float = 255.0) -> float:
    """Computes SSIM using skimage if available, otherwise pure NumPy implementation."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            ref,
            pred,
            win_size=7,
            channel_axis=-1 if ref.ndim == 3 else None,
            gaussian_weights=False,
            use_sample_covariance=True,
            data_range=data_range,
        ))
    except ImportError:
        # Wang et al. 2004 standard SSIM approximation
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ref = ref.astype(np.float64)
        pred = pred.astype(np.float64)
        mu1 = np.mean(ref)
        mu2 = np.mean(pred)
        sigma1_sq = np.var(ref)
        sigma2_sq = np.var(pred)
        sigma12 = np.mean((ref - mu1) * (pred - mu2))
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(ssim_val)


def compute_metrics_for_tile(pred_path: Path, ref_path: Path) -> dict:
    with Image.open(pred_path) as im_pred:
        pred_rgb = im_pred.convert("RGB")
    with Image.open(ref_path) as im_ref:
        ref_rgb = im_ref.convert("RGB")

    pred_arr = np.array(pred_rgb, dtype=np.float32)
    ref_arr = np.array(ref_rgb, dtype=np.float32)

    if pred_arr.shape != ref_arr.shape:
        raise ValueError(f"Shape mismatch: {pred_arr.shape} vs {ref_arr.shape}")

    psnr_val = compute_psnr_numpy(ref_arr, pred_arr, data_range=255.0)
    ssim_val = compute_ssim_numpy(ref_arr, pred_arr, data_range=255.0)

    return {
        "pred_path": str(pred_path),
        "ref_path": str(ref_path),
        "psnr_db": psnr_val,
        "ssim": ssim_val,
        "pred_min": float(np.min(pred_arr)),
        "pred_max": float(np.max(pred_arr)),
        "ref_min": float(np.min(ref_arr)),
        "ref_max": float(np.max(ref_arr)),
    }


def evaluate_predictions_directory(
    predictions_dir: Path,
    dataset_root: Path,
    output_csv: Path | None = None,
) -> dict:
    pred_dir = predictions_dir.expanduser().resolve()
    data_root = dataset_root.expanduser().resolve()
    naip_root = data_root / "naip"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
    if not naip_root.is_dir():
        raise FileNotFoundError(f"NAIP directory not found: {naip_root}")

    # Find prediction images (e.g. {tile_id}/sr.png or {tile_id}.png)
    pred_files = list(pred_dir.rglob("*.png"))
    pred_by_tile = {}
    for p in pred_files:
        tile_name = p.stem if p.stem != "sr" else p.parent.name
        pred_by_tile[tile_name] = p

    naip_chips = list(naip_root.rglob("rgb.png"))
    naip_by_tile = {p.parent.name: p for p in naip_chips}

    common_tiles = sorted(set(pred_by_tile.keys()) & set(naip_by_tile.keys()))
    print(f"Found {len(common_tiles)} common tiles to evaluate.")

    records = []
    psnr_list, ssim_list = [], []
    for tile in common_tiles:
        res = compute_metrics_for_tile(pred_by_tile[tile], naip_by_tile[tile])
        res["tile_id"] = tile
        records.append(res)
        psnr_list.append(res["psnr_db"])
        ssim_list.append(res["ssim"])

    summary = {
        "tiles_evaluated": len(records),
        "psnr_mean": float(statistics.mean(psnr_list)) if psnr_list else 0.0,
        "psnr_std": float(statistics.stdev(psnr_list)) if len(psnr_list) > 1 else 0.0,
        "ssim_mean": float(statistics.mean(ssim_list)) if ssim_list else 0.0,
        "ssim_std": float(statistics.stdev(ssim_list)) if len(ssim_list) > 1 else 0.0,
    }

    print("\n--- Validation Summary ---")
    print(f"Evaluated Tiles: {summary['tiles_evaluated']}")
    print(f"PSNR (dB):       {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
    print(f"SSIM:            {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile_id", "psnr_db", "ssim", "pred_min", "pred_max", "ref_min", "ref_max", "pred_path", "ref_path"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Detailed CSV exported to: {output_csv}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Path to directory containing super-resolved predictions")
    parser.add_argument("--dataset", type=Path, default=Path("data/small_val_set"), help="Path to dataset root containing naip/ folder")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    evaluate_predictions_directory(args.predictions, args.dataset, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
