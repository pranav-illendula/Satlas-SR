"""
EXP-011 Option B: Full 256-Tile Sentinel-2 Frame-Selection Evaluation Runner.

Evaluates 7 frame-selection strategies across all 256 validation tiles.
Strictly creates new predictions and evaluation metrics under:
  results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/
Preserves all existing baseline, checkpoint, and earlier experiment artifacts intact.
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
from scipy import stats

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssr.archs.rrdbnet_arch import SSR_RRDBNet
from ssr.metrics.cpsnr import calculate_cpsnr
from basicsr.metrics import calculate_psnr, calculate_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Paths
DATA_ROOT_S2 = REPO_ROOT.parent / "data" / "small_val_set" / "sentinel2"
DATA_ROOT_NAIP = REPO_ROOT.parent / "data" / "small_val_set" / "naip"
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "esrgan_8S2.pth"
MANIFEST_PATH = REPO_ROOT / "results" / "satlas_sr_full_validation" / "inference_manifest.csv"
FULL_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_011_frame_selection_audit" / "full_256_tiles"
PRED_DIR = FULL_DIR / "predictions"


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {arr.shape} for {path}")
        return arr


def get_strategies():
    return [
        {"id": "S0", "name": "Baseline Frozen (Seed 0)", "type": "Pseudorandom", "desc": "Reproduces frozen baseline random.seed(0) stream"},
        {"id": "S1", "name": "Alternate Seed (Seed 42)", "type": "Pseudorandom", "desc": "Alternative random.seed(42) stream"},
        {"id": "S2", "name": "Sorted Baseline (Permutation Isolation)", "type": "Content-Matched", "desc": "Sorted ascending permutation of S0 frames"},
        {"id": "S3", "name": "Uniform Strided Grid", "type": "Deterministic", "desc": "Fixed indices [0, 4, 8, 12, 16, 20, 24, 28]"},
        {"id": "S4", "name": "Early Window", "type": "Deterministic", "desc": "Fixed indices [0, 1, 2, 3, 4, 5, 6, 7]"},
        {"id": "S5", "name": "Late Window", "type": "Deterministic", "desc": "Fixed indices [24, 25, 26, 27, 28, 29, 30, 31]"},
        {"id": "S6", "name": "Low-Brightness Heuristic", "type": "Content-Heuristic", "desc": "Lowest 8 frames by mean RGB intensity, sorted ascending"},
    ]


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


def preflight_verification(tile_ids, baseline_manifest_rows):
    print("=" * 80)
    print("PREFLIGHT CHECKS & BASELINE SELECTION VERIFICATION")
    print("=" * 80)

    # 1. Verify tile count and sorting
    assert len(tile_ids) == 256, f"Expected 256 tiles, got {len(tile_ids)}"
    assert tile_ids == sorted(tile_ids), "Tile IDs must be sorted alphanumerically"
    print(f"PASS: 256 tiles verified in exact sorted order.")

    # 2. Verify S0 exact match against frozen baseline
    random.seed(0)
    simulated_s0 = {}
    for tile in tile_ids:
        simulated_s0[tile] = " ".join(str(x) for x in random.sample(range(32), 8))

    manifest_map = {r["tile_id"]: r["selected_frame_indices_zero_based"] for r in baseline_manifest_rows}
    matches = 0
    mismatches = []
    for tile in tile_ids:
        orig = manifest_map.get(tile)
        sim = simulated_s0.get(tile)
        if orig == sim:
            matches += 1
        else:
            mismatches.append((tile, orig, sim))

    print(f"PASS: S0 reproduction match against frozen baseline manifest: {matches}/256 ({matches/256*100:.1f}%)")
    if matches != 256:
        raise RuntimeError(f"S0 does NOT reproduce baseline exactly! {len(mismatches)} mismatches found.")

    # 3. Verify NAIP references exist for all 256 tiles
    naip_map = {}
    for p in DATA_ROOT_NAIP.rglob("rgb.png"):
        naip_map[p.parent.name] = p
    for tile in tile_ids:
        if tile not in naip_map:
            raise FileNotFoundError(f"Missing NAIP reference for tile {tile}")
    print(f"PASS: All 256 NAIP reference images verified on disk.")

    # 4. Verify S2 TCI files exist for all 256 tiles
    for tile in tile_ids:
        tci_path = DATA_ROOT_S2 / tile / "tci.png"
        if not tci_path.is_file():
            raise FileNotFoundError(f"Missing S2 TCI file: {tci_path}")
    print(f"PASS: All 256 Sentinel-2 TCI images verified on disk.")

    # 5. Verify Checkpoint existence
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")
    print(f"PASS: Checkpoint verified: {CHECKPOINT_PATH.name}")
    print("=" * 80)
    return naip_map


def main():
    start_total = time.time()
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load baseline manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        baseline_manifest_rows = list(csv.DictReader(f))
    tile_ids = [r["tile_id"] for r in baseline_manifest_rows]

    # Preflight verification
    naip_map = preflight_verification(tile_ids, baseline_manifest_rows)

    strategies = get_strategies()

    # 1. Compute frame selections for all strategies across 256 tiles
    print("\n--- Computing Frame Selections for All 7 Strategies across 256 Tiles ---")
    selections = {s["id"]: {} for s in strategies}

    # S0: Seed 0 sequential
    random.seed(0)
    for tile in tile_ids:
        selections["S0"][tile] = random.sample(range(32), 8)

    # S1: Seed 42 sequential
    random.seed(42)
    for tile in tile_ids:
        selections["S1"][tile] = random.sample(range(32), 8)

    # S2: Sorted S0 indices
    for tile in tile_ids:
        selections["S2"][tile] = sorted(selections["S0"][tile])

    # S3: Uniform strided
    for tile in tile_ids:
        selections["S3"][tile] = [0, 4, 8, 12, 16, 20, 24, 28]

    # S4: Early window
    for tile in tile_ids:
        selections["S4"][tile] = [0, 1, 2, 3, 4, 5, 6, 7]

    # S5: Late window
    for tile in tile_ids:
        selections["S5"][tile] = [24, 25, 26, 27, 28, 29, 30, 31]

    # S6: Low-brightness selection heuristic
    print("Computing S6 low-brightness heuristic for 256 tiles...")
    for idx, tile in enumerate(tile_ids, start=1):
        tci_path = DATA_ROOT_S2 / tile / "tci.png"
        img = skimage.io.imread(str(tci_path))
        chunks = np.reshape(img, (-1, 32, 32, 3))
        means = [float(np.mean(chunk)) for chunk in chunks]
        bright_sorted = sorted(range(32), key=lambda i: means[i])
        selections["S6"][tile] = sorted(bright_sorted[:8])
        if idx % 64 == 0 or idx == 256:
            print(f"  Processed {idx}/256 tiles for S6")

    # Save selections CSV
    selections_csv_path = FULL_DIR / "full_frame_selections.csv"
    with open(selections_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy_id", "strategy_name", "strategy_type", "tile_id", "frame_indices_ordered", "frame_indices_sorted"])
        for strat in strategies:
            s_id = strat["id"]
            for tile in tile_ids:
                indices = selections[s_id][tile]
                ordered_str = " ".join(str(x) for x in indices)
                sorted_str = " ".join(str(x) for x in sorted(indices))
                writer.writerow([s_id, strat["name"], strat["type"], tile, ordered_str, sorted_str])
    print(f"Saved full frame selections to {selections_csv_path}")

    # 2. Run Inference across 7 strategies x 256 tiles = 1,792 inferences
    print("\n--- Running Inference (1,792 inferences: 7 strategies x 256 tiles) ---")
    model = build_and_load_model(device)
    inference_start = time.time()

    for strat in strategies:
        s_id = strat["id"]
        strat_out_dir = PRED_DIR / s_id
        strat_out_dir.mkdir(parents=True, exist_ok=True)
        strat_start = time.time()

        for idx, tile in enumerate(tile_ids, start=1):
            tile_out_dir = strat_out_dir / tile
            tile_out_dir.mkdir(parents=True, exist_ok=True)
            sr_path = tile_out_dir / "sr.png"

            # Check if prediction already exists and is valid
            if sr_path.is_file() and os.path.getsize(sr_path) > 1000:
                continue

            tci_path = DATA_ROOT_S2 / tile / "tci.png"
            tci_img = skimage.io.imread(str(tci_path))
            chunks = np.reshape(tci_img, (-1, 32, 32, 3))

            indices = selections[s_id][tile]
            chosen_chunks = [chunks[i] for i in indices]

            tensor_chunks = [torch.as_tensor(c).permute(2, 0, 1) for c in chosen_chunks]
            input_tensor = torch.cat(tensor_chunks).unsqueeze(0).to(device).float() / 255.0

            with torch.inference_mode():
                pred = model(input_tensor)

            pred = torch.clamp(pred, 0, 1).squeeze().cpu().numpy()
            if pred.shape != (3, 128, 128):
                raise ValueError(f"Expected output (3, 128, 128), got {pred.shape}")
            pred_hwc = (np.transpose(pred, (1, 2, 0)) * 255.0).astype(np.uint8)

            skimage.io.imsave(str(sr_path), pred_hwc, check_contrast=False)

        print(f"Completed inference for {s_id} ({strat['name']}) in {time.time() - strat_start:.2f}s")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"All 1,792 inferences completed in {time.time() - inference_start:.2f}s.")

    # 3. Load Evaluators (LPIPS VGG + OpenCLIP ViT-bigG-14)
    print("\n--- Initializing Metric Evaluators (LPIPS VGG + OpenCLIP ViT-bigG-14) ---")
    import lpips
    lpips_vgg_fn = lpips.LPIPS(net="vgg").to(device).eval()

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14-CLIPA-336", pretrained="datacomp1b")
    clip_model = clip_model.to(device).eval()
    clip_img_size = (336, 336)
    print("Metric evaluators loaded successfully.")

    # 4. Precompute NAIP Reference Tensors & Embeddings
    print("\n--- Precomputing NAIP Reference Arrays and CLIP Embeddings (256 tiles) ---")
    ref_arrays = {}
    ref_tensors_lpips = {}
    ref_clip_embeds = {}

    eval_batch_size = 16
    for i in range(0, len(tile_ids), eval_batch_size):
        b_tiles = tile_ids[i:i + eval_batch_size]
        b_arrs = [load_image_rgb(naip_map[t]) for t in b_tiles]
        for t, arr in zip(b_tiles, b_arrs):
            ref_arrays[t] = arr

        b_tensors = [torch.as_tensor(arr).permute(2, 0, 1).float() / 255.0 for arr in b_arrs]
        b_tensor_stack = torch.stack(b_tensors).to(device)

        # Store for LPIPS
        for t, single_t in zip(b_tiles, b_tensor_stack):
            ref_tensors_lpips[t] = single_t.unsqueeze(0)

        # Compute CLIP embeddings
        with torch.no_grad():
            b_interp = F.interpolate(b_tensor_stack, clip_img_size)
            b_embeds = clip_model.encode_image(b_interp)
            for t, embed in zip(b_tiles, b_embeds):
                ref_clip_embeds[t] = embed.unsqueeze(0)

    print("Precomputed all 256 NAIP reference representations.")

    # 5. Evaluate Metrics for All 7 Strategies x 256 Tiles
    print("\n--- Computing Metrics for All 1,792 Predictions ---")
    metrics_by_strat = {s["id"]: {} for s in strategies}
    metric_rows = []

    eval_start = time.time()
    for s_idx, strat in enumerate(strategies, start=1):
        s_id = strat["id"]
        s_start = time.time()
        print(f"[{s_idx}/7] Evaluating {s_id} ({strat['name']})...")

        # Process in batches of 16 for GPU metrics
        for b_start in range(0, len(tile_ids), eval_batch_size):
            b_tiles = tile_ids[b_start:b_start + eval_batch_size]

            # Load prediction arrays
            b_preds = [load_image_rgb(PRED_DIR / s_id / t / "sr.png") for t in b_tiles]
            b_pred_tensors = torch.stack([torch.as_tensor(arr).permute(2, 0, 1).float() / 255.0 for arr in b_preds]).to(device)
            b_ref_tensors = torch.stack([ref_tensors_lpips[t].squeeze(0) for t in b_tiles]).to(device)

            # Batched LPIPS
            with torch.no_grad():
                b_lpips = lpips_vgg_fn(b_pred_tensors, b_ref_tensors).squeeze().cpu().numpy()
                if b_lpips.ndim == 0:
                    b_lpips = np.array([float(b_lpips)])

            # Batched CLIPScore
            with torch.no_grad():
                b_interp = F.interpolate(b_pred_tensors, clip_img_size)
                b_clip_embeds = clip_model.encode_image(b_interp)
                b_ref_embeds = torch.cat([ref_clip_embeds[t] for t in b_tiles], dim=0)
                b_clip_scores = F.cosine_similarity(b_clip_embeds, b_ref_embeds).cpu().numpy()
                if b_clip_scores.ndim == 0:
                    b_clip_scores = np.array([float(b_clip_scores)])

            # CPU metrics (cPSNR, raw PSNR, SSIM)
            for j, tile in enumerate(b_tiles):
                p_arr = b_preds[j]
                r_arr = ref_arrays[tile]

                cpsnr_c4 = float(calculate_cpsnr(p_arr, r_arr, crop_border=4, input_order="HWC", test_y_channel=False))
                psnr_raw = float(peak_signal_noise_ratio(r_arr, p_arr, data_range=255))
                ssim_c4 = float(calculate_ssim(p_arr, r_arr, crop_border=4, input_order="HWC", test_y_channel=False))
                lpips_val = float(b_lpips[j])
                clip_val = float(b_clip_scores[j])

                record = {
                    "strategy_id": s_id,
                    "strategy_name": strat["name"],
                    "strategy_type": strat["type"],
                    "tile_id": tile,
                    "cpsnr_crop4_db": round(cpsnr_c4, 4),
                    "psnr_raw_db": round(psnr_raw, 4),
                    "ssim_crop4": round(ssim_c4, 5),
                    "lpips_vgg": round(lpips_val, 5),
                    "clipscore": round(clip_val, 5),
                    "selected_frame_indices": " ".join(str(x) for x in selections[s_id][tile])
                }
                metrics_by_strat[s_id][tile] = record
                metric_rows.append(record)

        print(f"  Finished {s_id} in {time.time() - s_start:.2f}s")

    print(f"All 1,792 metric evaluations completed in {time.time() - eval_start:.2f}s.")

    # Save full per-tile metrics CSV
    per_tile_csv_path = FULL_DIR / "full_per_tile_metrics.csv"
    with open(per_tile_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["strategy_id", "strategy_name", "strategy_type", "tile_id", "cpsnr_crop4_db", "psnr_raw_db", "ssim_crop4", "lpips_vgg", "clipscore", "selected_frame_indices"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved full per-tile metrics to {per_tile_csv_path}")

    # 6. Statistical Analysis & Paired Significance Tests
    print("\n--- Performing Statistical Analysis & Paired Hypothesis Tests ---")
    s0_metrics = metrics_by_strat["S0"]
    test_rows = []
    metric_keys = [
        ("cpsnr_crop4_db", "cPSNR (crop 4, dB)", "higher"),
        ("psnr_raw_db", "PSNR (raw, dB)", "higher"),
        ("ssim_crop4", "SSIM (crop 4)", "higher"),
        ("lpips_vgg", "LPIPS (VGG)", "lower"),
        ("clipscore", "CLIPScore", "higher"),
    ]

    alt_strategies = [s for s in strategies if s["id"] != "S0"]
    n_alt = len(alt_strategies) # 6

    for strat in alt_strategies:
        s_id = strat["id"]
        s_metrics = metrics_by_strat[s_id]

        for m_key, m_label, direction in metric_keys:
            s0_vals = np.array([s0_metrics[t][m_key] for t in tile_ids])
            s_vals = np.array([s_metrics[t][m_key] for t in tile_ids])
            diffs = s_vals - s0_vals # diff = alt - baseline

            mean_d = float(np.mean(diffs))
            median_d = float(np.median(diffs))
            std_d = float(np.std(diffs, ddof=1))
            n = len(diffs)
            se_d = std_d / np.sqrt(n)
            # 95% Confidence Interval
            ci_crit = stats.t.ppf(0.975, df=n - 1)
            ci_lower = mean_d - ci_crit * se_d
            ci_upper = mean_d + ci_crit * se_d

            # Win/Loss/Tie
            if direction == "higher":
                improved = int(np.sum(diffs > 1e-6))
                worsened = int(np.sum(diffs < -1e-6))
            else: # lower is better
                improved = int(np.sum(diffs < -1e-6))
                worsened = int(np.sum(diffs > 1e-6))
            tied = n - improved - worsened

            # Paired Student's t-test
            t_res = stats.ttest_rel(s_vals, s0_vals)
            t_stat = float(t_res.statistic)
            t_pval = float(t_res.pvalue)

            # Wilcoxon signed-rank test
            try:
                w_res = stats.wilcoxon(diffs, zero_method="pratt")
                w_stat = float(w_res.statistic)
                w_pval = float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = np.nan, np.nan

            # Bonferroni adjusted p-value across 6 comparisons
            bonf_t_pval = min(1.0, t_pval * n_alt)
            bonf_w_pval = min(1.0, w_pval * n_alt)

            test_rows.append({
                "strategy_id": s_id,
                "strategy_name": strat["name"],
                "strategy_type": strat["type"],
                "metric_key": m_key,
                "metric_name": m_label,
                "direction": direction,
                "n_tiles": n,
                "baseline_mean": round(float(np.mean(s0_vals)), 4),
                "strategy_mean": round(float(np.mean(s_vals)), 4),
                "mean_delta": round(mean_d, 4),
                "median_delta": round(median_d, 4),
                "std_delta": round(std_d, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "tiles_improved": improved,
                "pct_improved": round(improved / n * 100, 1),
                "tiles_worsened": worsened,
                "pct_worsened": round(worsened / n * 100, 1),
                "tiles_tied": tied,
                "paired_t_statistic": round(t_stat, 4),
                "paired_t_pvalue": t_pval,
                "bonferroni_t_pvalue": bonf_t_pval,
                "wilcoxon_statistic": round(w_stat, 2),
                "wilcoxon_pvalue": w_pval,
                "bonferroni_wilcoxon_pvalue": bonf_w_pval,
            })

    test_csv_path = FULL_DIR / "paired_statistical_tests.csv"
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(test_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"Saved paired statistical tests to {test_csv_path}")

    # 7. Build Full Strategy Summary JSON
    print("\n--- Constructing Consolidated Summary JSON ---")
    summary = {
        "experiment": "EXP-011",
        "description": "Sentinel-2 Frame-Selection Sensitivity Full 256-Tile Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_tiles_evaluated": len(tile_ids),
        "checkpoint": str(CHECKPOINT_PATH),
        "total_runtime_seconds": round(time.time() - start_total, 2),
        "strategies": {}
    }

    for strat in strategies:
        s_id = strat["id"]
        s_dict = {
            "name": strat["name"],
            "type": strat["type"],
            "description": strat["desc"],
            "metrics": {}
        }
        for m_key, m_label, direction in metric_keys:
            vals = np.array([metrics_by_strat[s_id][t][m_key] for t in tile_ids])
            s_dict["metrics"][m_key] = {
                "label": m_label,
                "direction": direction,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        # If alternative strategy, add paired delta summary
        if s_id != "S0":
            s_dict["paired_deltas_vs_s0"] = {}
            for row in test_rows:
                if row["strategy_id"] == s_id:
                    s_dict["paired_deltas_vs_s0"][row["metric_key"]] = {
                        "mean_delta": row["mean_delta"],
                        "median_delta": row["median_delta"],
                        "ci_95": [row["ci_95_lower"], row["ci_95_upper"]],
                        "pct_improved": row["pct_improved"],
                        "pct_worsened": row["pct_worsened"],
                        "t_pvalue": row["paired_t_pvalue"],
                        "bonferroni_t_pvalue": row["bonferroni_t_pvalue"],
                        "wilcoxon_pvalue": row["wilcoxon_pvalue"],
                        "bonferroni_wilcoxon_pvalue": row["bonferroni_wilcoxon_pvalue"],
                    }

        summary["strategies"][s_id] = s_dict

    summary_json_path = FULL_DIR / "full_strategy_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full strategy summary JSON to {summary_json_path}")

    # Print summary table
    print("\n" + "=" * 115)
    print(f"{'Strat':<5} | {'Strategy Name':<38} | {'cPSNR (dB)':<11} | {'PSNR (dB)':<11} | {'SSIM':<8} | {'LPIPS':<8} | {'CLIPScore':<9}")
    print("-" * 115)
    for s_id, s_data in summary["strategies"].items():
        c = s_data["metrics"]["cpsnr_crop4_db"]["mean"]
        p = s_data["metrics"]["psnr_raw_db"]["mean"]
        s = s_data["metrics"]["ssim_crop4"]["mean"]
        l = s_data["metrics"]["lpips_vgg"]["mean"]
        cl = s_data["metrics"]["clipscore"]["mean"]
        print(f"{s_id:<5} | {s_data['name']:<38} | {c:<11.4f} | {p:<11.4f} | {s:<8.5f} | {l:<8.5f} | {cl:<9.5f}")
    print("=" * 115)
    print(f"Full 256-tile evaluation complete in {time.time() - start_total:.1f}s.")


if __name__ == "__main__":
    main()
