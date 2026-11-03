"""
EXP-011: Pilot Sentinel-2 Frame-Selection Sensitivity Runner.

Evaluates 7 frame-selection strategies across the 5 canonical validation tiles.
Strictly creates new predictions under results/satlas_sr_experiment_011_frame_selection_audit/pilot/
Preserves all existing baseline and previous experiment artifacts intact.
"""

import os
import sys
import time
import json
import csv
import random
from pathlib import Path

import numpy as np
import skimage.io
from PIL import Image
import torch
import torch.nn.functional as F

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssr.archs.rrdbnet_arch import SSR_RRDBNet
from ssr.metrics.cpsnr import calculate_cpsnr
from basicsr.metrics import calculate_psnr, calculate_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Define Canonical Pilot Tile IDs
PILOT_TILES = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]

DATA_ROOT_S2 = REPO_ROOT.parent / "data" / "small_val_set" / "sentinel2"
DATA_ROOT_NAIP = REPO_ROOT.parent / "data" / "small_val_set" / "naip"
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "esrgan_8S2.pth"
PILOT_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_011_frame_selection_audit" / "pilot"
PRED_DIR = PILOT_DIR / "predictions"


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {arr.shape} for {path}")
        return arr


def find_naip_ref(tile_id: str) -> Path:
    matches = list(DATA_ROOT_NAIP.rglob(f"{tile_id}/rgb.png"))
    if not matches:
        raise FileNotFoundError(f"Missing NAIP reference for tile {tile_id} in {DATA_ROOT_NAIP}")
    return matches[0]


def get_strategies():
    """Define the 7 candidate strategies."""
    return [
        {"id": "S0", "name": "Baseline Frozen (Seed 0)", "type": "Pseudorandom"},
        {"id": "S1", "name": "Alternate Seed (Seed 42)", "type": "Pseudorandom"},
        {"id": "S2", "name": "Sorted Baseline (Permutation Isolation)", "type": "Content-Matched"},
        {"id": "S3", "name": "Uniform Strided Grid", "type": "Deterministic"},
        {"id": "S4", "name": "Early Window", "type": "Deterministic"},
        {"id": "S5", "name": "Late Window", "type": "Deterministic"},
        {"id": "S6", "name": "Low-Brightness Heuristic (Cloud Avoidance)", "type": "Content-Heuristic"},
    ]


def compute_frame_selections(tile_ids, strategies):
    """
    Computes exact frame selections for each strategy and tile.
    Returns dict: selections[strategy_id][tile_id] = list_of_8_indices
    """
    selections = {s["id"]: {} for s in strategies}

    # S0: Baseline Seed 0 sequential stream
    random.seed(0)
    for tile in tile_ids:
        selections["S0"][tile] = random.sample(range(32), 8)

    # S1: Alternate Seed 42 sequential stream
    random.seed(42)
    for tile in tile_ids:
        selections["S1"][tile] = random.sample(range(32), 8)

    # S2: Sorted S0 indices (permutation test)
    for tile in tile_ids:
        selections["S2"][tile] = sorted(selections["S0"][tile])

    # S3: Uniform Strided Grid
    for tile in tile_ids:
        selections["S3"][tile] = [0, 4, 8, 12, 16, 20, 24, 28]

    # S4: Early Window
    for tile in tile_ids:
        selections["S4"][tile] = [0, 1, 2, 3, 4, 5, 6, 7]

    # S5: Late Window
    for tile in tile_ids:
        selections["S5"][tile] = [24, 25, 26, 27, 28, 29, 30, 31]

    # S6: Low-Brightness Heuristic (Cloud Avoidance)
    for tile in tile_ids:
        tci_path = DATA_ROOT_S2 / tile / "tci.png"
        img = skimage.io.imread(str(tci_path))
        chunks = np.reshape(img, (-1, 32, 32, 3))
        means = [float(np.mean(chunk)) for chunk in chunks]
        bright_sorted = sorted(range(32), key=lambda i: means[i])
        selections["S6"][tile] = sorted(bright_sorted[:8])

    return selections


def build_and_load_model(device):
    model = SSR_RRDBNet(
        num_in_ch=24,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    if "params_ema" not in checkpoint:
        raise KeyError(f"Expected params_ema in checkpoint; found keys: {list(checkpoint.keys())}")
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model = model.to(device).eval()
    return model


def main():
    print("=" * 80)
    print("EXP-011: Pilot Sentinel-2 Frame-Selection Sensitivity Runner")
    print("=" * 80)
    start_total = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    strategies = get_strategies()
    selections = compute_frame_selections(PILOT_TILES, strategies)

    # 1. Save frame selections CSV
    selections_csv_path = PILOT_DIR / "pilot_frame_selections.csv"
    with open(selections_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy_id", "strategy_name", "strategy_type", "tile_id", "frame_indices_ordered", "frame_indices_sorted"])
        for strat in strategies:
            s_id = strat["id"]
            for tile in PILOT_TILES:
                indices = selections[s_id][tile]
                ordered_str = " ".join(str(x) for x in indices)
                sorted_str = " ".join(str(x) for x in sorted(indices))
                writer.writerow([s_id, strat["name"], strat["type"], tile, ordered_str, sorted_str])
    print(f"Saved frame selections to {selections_csv_path}")

    # 2. Run Inference
    print("\n--- Running Inference Across 7 Strategies x 5 Tiles (35 inferences) ---")
    model = build_and_load_model(device)

    inference_log_rows = []
    
    for strat in strategies:
        s_id = strat["id"]
        strat_out_dir = PRED_DIR / s_id
        strat_out_dir.mkdir(parents=True, exist_ok=True)

        for tile in PILOT_TILES:
            t0 = time.time()
            tile_out_dir = strat_out_dir / tile
            tile_out_dir.mkdir(parents=True, exist_ok=True)
            sr_path = tile_out_dir / "sr.png"
            lr_path = tile_out_dir / "lr.png"

            tci_path = DATA_ROOT_S2 / tile / "tci.png"
            tci_img = skimage.io.imread(str(tci_path))
            chunks = np.reshape(tci_img, (-1, 32, 32, 3))

            indices = selections[s_id][tile]
            chosen_chunks = [chunks[i] for i in indices]

            # Reference LR image: first frame of the selection
            lr_img = chosen_chunks[0]

            # Convert to PyTorch tensor [1, 24, 32, 32] normalized by 255
            tensor_chunks = [torch.as_tensor(c).permute(2, 0, 1) for c in chosen_chunks]
            input_tensor = torch.cat(tensor_chunks).unsqueeze(0).to(device).float() / 255.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                pred = model(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_mib = 0.0

            pred = torch.clamp(pred, 0, 1).squeeze().cpu().numpy()
            if pred.shape != (3, 128, 128):
                raise ValueError(f"Expected output (3, 128, 128), got {pred.shape}")
            pred_hwc = (np.transpose(pred, (1, 2, 0)) * 255.0).astype(np.uint8)

            skimage.io.imsave(str(lr_path), lr_img)
            skimage.io.imsave(str(sr_path), pred_hwc, check_contrast=False)

            elapsed = time.time() - t0
            inference_log_rows.append({
                "strategy_id": s_id,
                "strategy_name": strat["name"],
                "tile_id": tile,
                "status": "OK",
                "runtime_seconds": f"{elapsed:.4f}",
                "gpu_peak_allocated_mib": f"{peak_mib:.2f}",
                "sr_output_path": str(sr_path),
                "sr_shape": f"{pred_hwc.shape[0]}x{pred_hwc.shape[1]}"
            })
            print(f"[{s_id}] Tile {tile}: {pred_hwc.shape} saved in {elapsed:.3f}s (Peak VRAM: {peak_mib:.1f} MiB)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Load Metric Evaluators
    print("\n--- Initializing Metric Evaluators (LPIPS VGG + OpenCLIP ViT-bigG-14) ---")
    import lpips
    lpips_vgg_fn = lpips.LPIPS(net="vgg").to(device).eval()

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14-CLIPA-336", pretrained="datacomp1b")
    clip_model = clip_model.to(device).eval()
    clip_img_size = (336, 336)
    print("Metric evaluators initialized successfully.")

    # 4. Compute Metrics
    print("\n--- Computing Metrics for All 35 Predictions Against NAIP ---")
    metric_rows = []

    for strat in strategies:
        s_id = strat["id"]
        for tile in PILOT_TILES:
            sr_path = PRED_DIR / s_id / tile / "sr.png"
            naip_path = find_naip_ref(tile)

            pred_arr = load_image_rgb(sr_path)
            ref_arr = load_image_rgb(naip_path)

            if pred_arr.shape != (128, 128, 3) or ref_arr.shape != (128, 128, 3):
                raise ValueError(f"Shape mismatch: {pred_arr.shape} vs {ref_arr.shape}")

            # cPSNR
            cpsnr_c4 = float(calculate_cpsnr(pred_arr, ref_arr, crop_border=4, input_order="HWC", test_y_channel=False))
            cpsnr_c0 = float(calculate_cpsnr(pred_arr, ref_arr, crop_border=0, input_order="HWC", test_y_channel=False))

            # PSNR & SSIM crop 4
            psnr_c4 = float(calculate_psnr(pred_arr, ref_arr, crop_border=4, input_order="HWC", test_y_channel=False))
            ssim_c4 = float(calculate_ssim(pred_arr, ref_arr, crop_border=4, input_order="HWC", test_y_channel=False))

            # PSNR & SSIM raw
            psnr_raw = float(peak_signal_noise_ratio(ref_arr, pred_arr, data_range=255))
            ssim_raw = float(structural_similarity(ref_arr, pred_arr, channel_axis=2, data_range=255))

            # LPIPS
            t_pred = torch.as_tensor(pred_arr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            t_ref = torch.as_tensor(ref_arr).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            with torch.no_grad():
                lpips_val = float(lpips_vgg_fn(t_pred, t_ref).detach().item())

            # CLIPScore
            with torch.no_grad():
                t1_interp = F.interpolate(t_pred, clip_img_size)
                t2_interp = F.interpolate(t_ref, clip_img_size)
                f1 = clip_model.encode_image(t1_interp)
                f2 = clip_model.encode_image(t2_interp)
                c_score = float(F.cosine_similarity(f1, f2).detach().item())

            indices_str = " ".join(str(x) for x in selections[s_id][tile])

            metric_rows.append({
                "strategy_id": s_id,
                "strategy_name": strat["name"],
                "strategy_type": strat["type"],
                "tile_id": tile,
                "selected_frame_indices": indices_str,
                "cpsnr_crop4_db": round(cpsnr_c4, 4),
                "cpsnr_crop0_db": round(cpsnr_c0, 4),
                "psnr_crop4_db": round(psnr_c4, 4),
                "ssim_crop4": round(ssim_c4, 5),
                "psnr_raw_db": round(psnr_raw, 4),
                "ssim_raw": round(ssim_raw, 5),
                "lpips_vgg": round(lpips_val, 5),
                "clipscore": round(c_score, 5),
                "prediction_path": str(sr_path),
                "reference_path": str(naip_path)
            })

    # Save per-tile metrics CSV
    per_tile_csv_path = PILOT_DIR / "pilot_per_tile_metrics.csv"
    with open(per_tile_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(metric_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved per-tile metrics to {per_tile_csv_path}")

    # 5. Compute Aggregates and Summary
    print("\n--- Computing Aggregated Strategy Statistics & Deltas ---")
    summary = {
        "experiment": "EXP-011",
        "description": "Sentinel-2 Frame-Selection Sensitivity Pilot (5 Tiles)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tiles_evaluated": PILOT_TILES,
        "checkpoint_used": str(CHECKPOINT_PATH),
        "total_runtime_seconds": round(time.time() - start_total, 2),
        "strategies": {}
    }

    # Group metrics by strategy
    by_strat = {s["id"]: [] for s in strategies}
    for r in metric_rows:
        by_strat[r["strategy_id"]].append(r)

    # Base S0 values per tile for delta computation
    base_by_tile = {r["tile_id"]: r for r in by_strat["S0"]}

    for strat in strategies:
        s_id = strat["id"]
        rows = by_strat[s_id]

        def get_stat(key):
            vals = [r[key] for r in rows]
            return {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        def get_delta_stat(key):
            deltas = [r[key] - base_by_tile[r["tile_id"]][key] for r in rows]
            return {
                "mean_delta": round(float(np.mean(deltas)), 4),
                "std_delta": round(float(np.std(deltas, ddof=1)), 4),
                "min_delta": round(float(np.min(deltas)), 4),
                "max_delta": round(float(np.max(deltas)), 4),
            }

        stat_dict = {
            "name": strat["name"],
            "type": strat["type"],
            "cpsnr_crop4_db": get_stat("cpsnr_crop4_db"),
            "psnr_crop4_db": get_stat("psnr_crop4_db"),
            "ssim_crop4": get_stat("ssim_crop4"),
            "lpips_vgg": get_stat("lpips_vgg"),
            "clipscore": get_stat("clipscore"),
        }

        if s_id != "S0":
            stat_dict["delta_vs_baseline_s0"] = {
                "cpsnr_crop4_db": get_delta_stat("cpsnr_crop4_db"),
                "psnr_crop4_db": get_delta_stat("psnr_crop4_db"),
                "ssim_crop4": get_delta_stat("ssim_crop4"),
                "lpips_vgg": get_delta_stat("lpips_vgg"),
                "clipscore": get_delta_stat("clipscore"),
            }

        summary["strategies"][s_id] = stat_dict

    summary_json_path = PILOT_DIR / "pilot_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pilot summary JSON to {summary_json_path}")

    print("\n--- Pilot Results Summary Table ---")
    print(f"{'Strat':<5} | {'Strategy Name':<38} | {'cPSNR (dB)':<11} | {'PSNR (dB)':<11} | {'SSIM':<8} | {'LPIPS':<8} | {'CLIPScore':<9}")
    print("-" * 105)
    for s_id, s_data in summary["strategies"].items():
        c = s_data["cpsnr_crop4_db"]["mean"]
        p = s_data["psnr_crop4_db"]["mean"]
        s = s_data["ssim_crop4"]["mean"]
        l = s_data["lpips_vgg"]["mean"]
        cl = s_data["clipscore"]["mean"]
        print(f"{s_id:<5} | {s_data['name']:<38} | {c:<11.4f} | {p:<11.4f} | {s:<8.5f} | {l:<8.5f} | {cl:<9.5f}")

    print("=" * 80)
    print("Pilot completed successfully.")


if __name__ == "__main__":
    main()
