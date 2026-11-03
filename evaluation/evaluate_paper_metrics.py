#!/usr/bin/env python3
"""
Official research benchmark evaluator for Satlas-SR super-resolution predictions.
Evaluates:
  1. cPSNR (PROBA-V translation & brightness bias compensated PSNR, crop_border=4)
  2. LPIPS (VGG16 perceptual distance)
  3. CLIPScore (OpenCLIP ViT-bigG-14-CLIPA-336 feature cosine similarity)
  4. Standard PSNR & SSIM (crop_border=4)
Strictly read-only; never modifies predictions or ground truth reference data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image

from evaluation.metrics.cpsnr import calculate_cpsnr
from evaluation.evaluate_validation import compute_psnr_numpy, compute_ssim_numpy


def evaluate_paper_metrics(
    predictions_dir: Path,
    dataset_root: Path,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    compute_deep_metrics: bool = True,
    device: str = "cuda",
) -> dict:
    pred_dir = predictions_dir.expanduser().resolve()
    data_root = dataset_root.expanduser().resolve()
    naip_root = data_root / "naip"

    if not pred_dir.is_dir() or not naip_root.is_dir():
        raise FileNotFoundError(f"Missing predictions or dataset directory: {pred_dir}, {naip_root}")

    # Gather prediction images
    pred_files = list(pred_dir.rglob("*.png"))
    pred_map = {}
    for p in pred_files:
        tile = p.stem if p.stem != "sr" else p.parent.name
        pred_map[tile] = p

    naip_chips = list(naip_root.rglob("rgb.png"))
    naip_map = {p.parent.name: p for p in naip_chips}

    common_tiles = sorted(set(pred_map.keys()) & set(naip_map.keys()))
    print(f"Evaluating {len(common_tiles)} common tiles with official paper benchmark protocol.")

    # Lazy load deep metrics
    lpips_fn = None
    clip_fn = None
    if compute_deep_metrics:
        try:
            from evaluation.metrics.lpips import calculate_lpips
            lpips_fn = calculate_lpips
        except ImportError as e:
            print(f"Warning: LPIPS disabled ({e})")

        try:
            from evaluation.metrics.clipscore import calculate_clipscore
            clip_fn = calculate_clipscore
        except ImportError as e:
            print(f"Warning: CLIPScore disabled ({e})")

    records = []
    cpsnr_list, psnr_list, ssim_list, lpips_list, clip_list = [], [], [], [], []

    for idx, tile in enumerate(common_tiles, 1):
        with Image.open(pred_map[tile]) as im_p:
            pred_rgb = im_p.convert("RGB")
        with Image.open(naip_map[tile]) as im_r:
            ref_rgb = im_r.convert("RGB")

        p_arr = np.array(pred_rgb)
        r_arr = np.array(ref_rgb)

        # 1. cPSNR (official PROBA-V metric, crop_border=4)
        cpsnr_val = calculate_cpsnr(p_arr, r_arr, crop_border=4, max_offset=8)
        cpsnr_list.append(cpsnr_val)

        # 2. Standard PSNR & SSIM (crop_border=4)
        p_crop = p_arr[4:-4, 4:-4, :]
        r_crop = r_arr[4:-4, 4:-4, :]
        psnr_val = compute_psnr_numpy(r_crop, p_crop, data_range=255.0)
        ssim_val = compute_ssim_numpy(r_crop, p_crop, data_range=255.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        row = {
            "tile_id": tile,
            "cpsnr_crop4_db": cpsnr_val,
            "psnr_crop4_db": psnr_val,
            "cpsnr_gain_db": cpsnr_val - psnr_val,
            "ssim_crop4": ssim_val,
        }

        # 3. LPIPS
        if lpips_fn:
            try:
                l_val = lpips_fn(p_arr, r_arr, lpips_model="vgg", device=device)
                row["lpips_vgg"] = l_val
                lpips_list.append(l_val)
            except Exception:
                row["lpips_vgg"] = None

        # 4. CLIPScore
        if clip_fn:
            try:
                c_val = clip_fn(p_arr, r_arr, clip_model="clipa-ViT-bigG-14", device=device)
                row["clipscore_vit_bigg"] = c_val
                clip_list.append(c_val)
            except Exception:
                row["clipscore_vit_bigg"] = None

        records.append(row)
        if idx % 50 == 0 or idx == len(common_tiles):
            print(f"Processed {idx}/{len(common_tiles)} tiles...")

    summary = {
        "tiles_evaluated": len(records),
        "cpsnr_mean": float(np.mean(cpsnr_list)),
        "cpsnr_std": float(np.std(cpsnr_list)),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "cpsnr_gain_mean": float(np.mean(np.array(cpsnr_list) - np.array(psnr_list))),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
    }
    if lpips_list:
        summary["lpips_mean"] = float(np.mean(lpips_list))
        summary["lpips_std"] = float(np.std(lpips_list))
    if clip_list:
        summary["clipscore_mean"] = float(np.mean(clip_list))
        summary["clipscore_std"] = float(np.std(clip_list))

    print("\n" + "=" * 50)
    print("      SATLAS-SR OFFICIAL BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Tiles Evaluated:  {summary['tiles_evaluated']}")
    print(f"cPSNR (crop 4):   {summary['cpsnr_mean']:.2f} ± {summary['cpsnr_std']:.2f} dB")
    print(f"PSNR (crop 4):    {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")
    print(f"cPSNR Gain:       +{summary['cpsnr_gain_mean']:.2f} dB")
    print(f"SSIM (crop 4):    {summary['ssim_mean']:.3f} ± {summary['ssim_std']:.3f}")
    if "lpips_mean" in summary:
        print(f"LPIPS (VGG16):    {summary['lpips_mean']:.3f} ± {summary['lpips_std']:.3f}")
    if "clipscore_mean" in summary:
        print(f"CLIPScore:        {summary['clipscore_mean']:.3f} ± {summary['clipscore_std']:.3f}")
    print("=" * 50)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"summary": summary, "per_tile": records}, f, indent=2)
        print(f"Benchmark results JSON saved to: {output_json}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Benchmark results CSV saved to: {output_csv}")

    return summary


def main() -> int:
    default_pred = repo_root / "results" / "satlas_sr_pilot_five_tiles"
    default_data = repo_root / "data" / "small_val_set"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=default_pred, help=f"Directory containing super-resolved predictions (default: {default_pred})")
    parser.add_argument("--dataset", type=Path, default=default_data, help=f"Dataset directory containing naip/ (default: {default_data})")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--no-deep", action="store_true", help="Skip LPIPS and CLIPScore models")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    evaluate_paper_metrics(
        args.predictions,
        args.dataset,
        output_json=args.output_json,
        output_csv=args.output_csv,
        compute_deep_metrics=not args.no_deep,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
