import os
import sys
import time
import json
import csv
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(".").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import scipy.stats as stats
from scipy.signal import convolve2d
import skimage.io

from ssr.archs.rrdbnet_arch import SSR_RRDBNet

CHECKPOINT_PATH = Path("experiments/esrgan_8S2.pth")
DATA_ROOT_S2 = Path("../data/small_val_set/sentinel2")
DATA_ROOT_NAIP = Path("../data/small_val_set/naip")
SPLIT_MANIFEST_PATH = Path("results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json")
INFERENCE_MANIFEST_PATH = Path("results/satlas_sr_full_validation/inference_manifest.csv")
EXP014_DIR = Path("results/satlas_sr_experiment_014_tta_uncertainty")
MAPS_DIR = EXP014_DIR / "maps"

def apply_d4_forward(x: torch.Tensor, k: int, f: int) -> torch.Tensor:
    out = x
    if f == 1:
        out = torch.flip(out, dims=[-1])
    if k != 0:
        out = torch.rot90(out, k=k, dims=[-2, -1])
    return out

def apply_d4_inverse(x: torch.Tensor, k: int, f: int) -> torch.Tensor:
    out = x
    if k != 0:
        inv_k = (4 - k) % 4
        out = torch.rot90(out, k=inv_k, dims=[-2, -1])
    if f == 1:
        out = torch.flip(out, dims=[-1])
    return out

def compute_registration_shift(pred: np.ndarray, ref: np.ndarray, max_shift: int = 8, border: int = 4):
    """
    Find integer translation (dx, dy) in [-max_shift, max_shift] minimizing MSE on interior crop.
    pred, ref: (H, W, C) float in [0, 1]
    """
    H, W, C = pred.shape
    pred_crop = pred[border:H-border, border:W-border]
    cH, cW, _ = pred_crop.shape
    
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
            mse = np.mean((pred_crop - ref_crop) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_dx = dx
                best_dy = dy
                
    return best_dx, best_dy, best_mse

def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    """Compute normalized 2D Sobel gradient magnitude on luminance of (H, W, C)."""
    lum = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = convolve2d(lum, gx, mode="same", boundary="symm")
    grad_y = convolve2d(lum, gy, mode="same", boundary="symm")
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return mag

def compute_sparsification_metrics(errors: np.ndarray, uncertainties: np.ndarray, fractions=None):
    """
    errors, uncertainties: 1D flat arrays of length N
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.95, 20)
    
    N = len(errors)
    sort_idx = np.argsort(-uncertainties)
    sorted_errors = errors[sort_idx]
    
    oracle_sort_idx = np.argsort(-errors)
    oracle_errors = errors[oracle_sort_idx]
    
    curve_model = []
    curve_oracle = []
    curve_random = []
    mae_initial = float(np.mean(errors))
    
    for p in fractions:
        k = int(p * N)
        if k >= N:
            break
        rem_m = sorted_errors[k:]
        curve_model.append(float(np.mean(rem_m)) if len(rem_m) > 0 else 0.0)
        
        rem_o = oracle_errors[k:]
        curve_oracle.append(float(np.mean(rem_o)) if len(rem_o) > 0 else 0.0)
        
        curve_random.append(mae_initial)
        
    curve_model = np.array(curve_model)
    curve_oracle = np.array(curve_oracle)
    curve_random = np.array(curve_random)
    
    p_steps = fractions[:len(curve_model)]
    ause = float(np.trapz(curve_model - curve_oracle, p_steps))
    ause_random = float(np.trapz(curve_random - curve_oracle, p_steps))
    aurg = float(ause_random - ause)
    
    # MAE reduction at 20% removal (p=0.20)
    idx_20 = int(0.20 * len(curve_model))
    mae_at_20 = float(curve_model[idx_20]) if idx_20 < len(curve_model) else mae_initial
    pct_red_20 = float((mae_initial - mae_at_20) / mae_initial * 100.0) if mae_initial > 0 else 0.0

    return {
        "ause": ause,
        "ause_random": ause_random,
        "aurg": aurg,
        "mae_initial": mae_initial,
        "mae_at_20pct_removal": mae_at_20,
        "pct_reduction_at_20pct": pct_red_20
    }

def main():
    start_total = time.time()
    print("=" * 80)
    print("EXP-014: FULL DEVELOPMENT SPLIT EVALUATION (107 TILES)")
    print("=" * 80)
    
    EXP014_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load split manifest
    with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    dev_tiles = split_data["development_tiles"]
    holdout_tiles = split_data["holdout_tiles"]
    quarantine_tiles = split_data["quarantined_tiles"]
    
    assert len(dev_tiles) == 107, f"Expected 107 dev tiles, found {len(dev_tiles)}"
    print(f"Loaded {len(dev_tiles)} development tiles.")
    print(f"Holdout ({len(holdout_tiles)} tiles) and Quarantine ({len(quarantine_tiles)} tiles) are STRICTLY LOCKED.")

    # 2. Load inference manifest for S0 frame selections and NAIP paths
    with open(INFERENCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    
    tile_indices_map = {r["tile_id"]: [int(x) for x in r["selected_frame_indices_zero_based"].split()] for r in manifest_rows}
    tile_naip_path_map = {r["tile_id"]: Path(r["naip_reference_path"]) for r in manifest_rows}

    # 3. Build model
    model = SSR_RRDBNet(
        num_in_ch=24,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["params_ema"], strict=True)
    model = model.to(device).eval()
    print("Model loaded successfully.")

    # D4 transformations: 8 symmetries
    d4_transforms = [(k, f) for f in [0, 1] for k in [0, 1, 2, 3]]
    border = 4  # cPSNR standard border exclusion

    per_tile_records = []
    np.random.seed(42)  # For reproducible random baseline

    print(f"\nBeginning evaluation across {len(dev_tiles)} development tiles...")

    for idx, tile in enumerate(dev_tiles, start=1):
        tile_start = time.time()
        
        # Load Sentinel-2 TCI
        tci_path = DATA_ROOT_S2 / tile / "tci.png"
        tci_img = skimage.io.imread(str(tci_path))
        chunks = np.reshape(tci_img, (-1, 32, 32, 3))
        
        indices = tile_indices_map[tile]
        chosen_chunks = [chunks[i] for i in indices]
        tensor_chunks = [torch.as_tensor(c).permute(2, 0, 1) for c in chosen_chunks]
        input_tensor = torch.cat(tensor_chunks).unsqueeze(0).to(device).float() / 255.0

        # Run 8 D4 transformations
        transformed_preds = []
        for (k, f) in d4_transforms:
            x_fwd = apply_d4_forward(input_tensor, k, f)
            with torch.inference_mode():
                y_fwd = model(x_fwd)
            y_inv = apply_d4_inverse(y_fwd, k, f)
            y_inv_np = torch.clamp(y_inv, 0, 1).squeeze().cpu().numpy() # (3, 128, 128)
            transformed_preds.append(y_inv_np)

        stack = np.stack(transformed_preds, axis=0) # (8, 3, 128, 128)
        
        # Mean prediction across 8 passes
        mean_pred = np.mean(stack, axis=0) # (3, 128, 128)
        mean_pred_hwc = np.transpose(mean_pred, (1, 2, 0)) # (128, 128, 3)
        
        # Baseline T0 prediction (single forward pass)
        base_pred = stack[0] # (3, 128, 128)
        base_pred_hwc = np.transpose(base_pred, (1, 2, 0)) # (128, 128, 3)

        # Per-pixel variance and std across 8 passes
        pixel_var_c = np.var(stack, axis=0, ddof=1) # (3, 128, 128)
        pixel_var = np.mean(pixel_var_c, axis=0) # (128, 128)
        pixel_std = np.sqrt(pixel_var) # (128, 128) - Scalar Uncertainty U_TTA

        # Load NAIP reference
        naip_path = tile_naip_path_map[tile]
        naip_img = skimage.io.imread(str(naip_path)).astype(np.float32) / 255.0 # (128, 128, 3)

        # Baseline 1: Constant random uncertainty
        u_random = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)

        # Baseline 2: Sobel gradient magnitude of base prediction
        u_gradient = compute_sobel_gradient(base_pred_hwc)

        # Registration Shift Optimization on interior crop (border=4)
        best_dx, best_dy, best_mse = compute_registration_shift(base_pred_hwc, naip_img, max_shift=8, border=border)
        shift_mag = float(np.sqrt(best_dx ** 2 + best_dy ** 2))

        # Compute full error maps
        # Uncompensated absolute error
        err_uncomp_full = np.mean(np.abs(base_pred_hwc - naip_img), axis=2) # (128, 128)

        # Compensated reference crop
        cH, cW = 128 - 2 * border, 128 - 2 * border
        y_ref_start = border + best_dy
        x_ref_start = border + best_dx
        ref_comp_crop = naip_img[y_ref_start:y_ref_start+cH, x_ref_start:x_ref_start+cW]
        base_pred_crop = base_pred_hwc[border:128-border, border:128-border]
        
        err_comp_crop = np.mean(np.abs(base_pred_crop - ref_comp_crop), axis=2) # (120, 120)
        err_uncomp_crop = err_uncomp_full[border:128-border, border:128-border] # (120, 120)

        # Crop uncertainty maps to matching interior window
        u_tta_crop = pixel_std[border:128-border, border:128-border]
        u_grad_crop = u_gradient[border:128-border, border:128-border]
        u_rand_crop = u_random[border:128-border, border:128-border]

        # Flatten interior arrays for statistical ranking
        flat_err_comp = err_comp_crop.flatten()
        flat_err_uncomp = err_uncomp_crop.flatten()
        flat_u_tta = u_tta_crop.flatten()
        flat_u_grad = u_grad_crop.flatten()
        flat_u_rand = u_rand_crop.flatten()

        # Spearman rank correlations
        spearman_comp, p_comp = stats.spearmanr(flat_u_tta, flat_err_comp)
        spearman_uncomp, p_uncomp = stats.spearmanr(flat_u_tta, flat_err_uncomp)
        spearman_grad, p_grad = stats.spearmanr(flat_u_grad, flat_err_comp)
        spearman_rand, p_rand = stats.spearmanr(flat_u_rand, flat_err_comp)

        # Sparsification analysis on compensated error
        sp_tta = compute_sparsification_metrics(flat_err_comp, flat_u_tta)
        sp_grad = compute_sparsification_metrics(flat_err_comp, flat_u_grad)
        sp_rand = compute_sparsification_metrics(flat_err_comp, flat_u_rand)

        # Edge stratification on reference
        ref_grad_crop = compute_sobel_gradient(ref_comp_crop)
        edge_thresh = np.percentile(ref_grad_crop, 85) # Top 15% gradient
        edge_mask = ref_grad_crop >= edge_thresh
        
        if np.sum(edge_mask) > 10 and np.sum(~edge_mask) > 10:
            spearman_edge, _ = stats.spearmanr(flat_u_tta[edge_mask.flatten()], flat_err_comp[edge_mask.flatten()])
            spearman_bg, _ = stats.spearmanr(flat_u_tta[(~edge_mask).flatten()], flat_err_comp[(~edge_mask).flatten()])
        else:
            spearman_edge, spearman_bg = float("nan"), float("nan")

        # Save map artifact for this tile
        map_path = MAPS_DIR / f"{tile}_uncertainty.npz"
        np.savez_compressed(
            str(map_path),
            u_tta=pixel_std.astype(np.float32),
            u_var=pixel_var.astype(np.float32),
            err_comp=err_comp_crop.astype(np.float32),
            err_uncomp=err_uncomp_crop.astype(np.float32),
            shift_dx=best_dx,
            shift_dy=best_dy
        )

        record = {
            "tile_id": tile,
            "reg_dx_px": best_dx,
            "reg_dy_px": best_dy,
            "reg_shift_mag_px": round(shift_mag, 3),
            "mae_comp": round(sp_tta["mae_initial"], 6),
            "mae_uncomp": round(float(np.mean(flat_err_uncomp)), 6),
            "mean_uncertainty_std": round(float(np.mean(flat_u_tta)), 6),
            "max_uncertainty_std": round(float(np.max(flat_u_tta)), 6),
            "spearman_rho_comp": round(float(spearman_comp), 6),
            "spearman_p_comp": float(p_comp),
            "spearman_rho_uncomp": round(float(spearman_uncomp), 6),
            "spearman_p_uncomp": float(p_uncomp),
            "spearman_rho_grad": round(float(spearman_grad), 6),
            "spearman_rho_rand": round(float(spearman_rand), 6),
            "ause_tta": round(sp_tta["ause"], 6),
            "ause_random": round(sp_tta["ause_random"], 6),
            "aurg_tta": round(sp_tta["aurg"], 6),
            "ause_grad": round(sp_grad["ause"], 6),
            "aurg_grad": round(sp_grad["aurg"], 6),
            "pct_mae_reduction_20pct": round(sp_tta["pct_reduction_at_20pct"], 3),
            "spearman_edge": round(float(spearman_edge), 6) if not np.isnan(spearman_edge) else None,
            "spearman_bg": round(float(spearman_bg), 6) if not np.isnan(spearman_bg) else None
        }
        per_tile_records.append(record)

        if idx % 10 == 0 or idx == len(dev_tiles):
            elapsed = time.time() - start_total
            print(f"[{idx:>3}/{len(dev_tiles)}] Processed {tile} | rho_comp={spearman_comp:+.4f} | aurg={sp_tta['aurg']:+.4f} | elapsed={elapsed:.1f}s")

    # 4. Save per-tile CSV
    csv_path = EXP014_DIR / "per_tile_uncertainty_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tile_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_tile_records)
    print(f"\nSaved per-tile metrics to {csv_path}")

    # 5. Compute Aggregate Summary Statistics
    rhos_comp = [r["spearman_rho_comp"] for r in per_tile_records]
    rhos_uncomp = [r["spearman_rho_uncomp"] for r in per_tile_records]
    rhos_grad = [r["spearman_rho_grad"] for r in per_tile_records]
    rhos_rand = [r["spearman_rho_rand"] for r in per_tile_records]
    aurgs_tta = [r["aurg_tta"] for r in per_tile_records]
    aurgs_grad = [r["aurg_grad"] for r in per_tile_records]
    pct_reds_20 = [r["pct_mae_reduction_20pct"] for r in per_tile_records]
    shifts = [r["reg_shift_mag_px"] for r in per_tile_records]
    
    # Paired statistical tests: TTA vs Random, TTA vs Gradient
    # Wilcoxon signed rank test for AURG > 0 (TTA vs Random)
    w_aurg, p_aurg = stats.wilcoxon(aurgs_tta, alternative="greater")
    
    # Paired t-test and Wilcoxon for Spearman rho: TTA vs Gradient
    t_rho_grad, p_rho_grad_t = stats.ttest_rel(rhos_comp, rhos_grad)
    w_rho_grad, p_rho_grad_w = stats.wilcoxon(rhos_comp, rhos_grad)

    def stats_dict(vals):
        n = len(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        sem = sd / np.sqrt(n)
        ci95_half = 1.9826 * sem  # t_0.975 for df=106
        return {
            "mean": round(mean, 6),
            "std": round(sd, 6),
            "sem": round(sem, 6),
            "ci95": [round(mean - ci95_half, 6), round(mean + ci95_half, 6)],
            "median": round(float(np.median(vals)), 6),
            "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6)
        }

    summary = {
        "experiment_id": "EXP-014",
        "title": "D4 Test-Time Augmentation Uncertainty Prototype Development Evaluation",
        "timestamp": "2026-10-23T15:20:00+05:30",
        "evaluated_partition": "development_split_only",
        "tile_count": len(dev_tiles),
        "holdout_status": "LOCKED_UNTOUCHED",
        "quarantine_status": "LOCKED_EXCLUDED",
        "runtime_total_seconds": round(time.time() - start_total, 2),
        "registration_shifts": stats_dict(shifts),
        "spearman_rho": {
            "tta_compensated": stats_dict(rhos_comp),
            "tta_uncompensated": stats_dict(rhos_uncomp),
            "gradient_baseline": stats_dict(rhos_grad),
            "random_baseline": stats_dict(rhos_rand)
        },
        "area_under_random_gain_aurg": {
            "tta_aurg": stats_dict(aurgs_tta),
            "gradient_aurg": stats_dict(aurgs_grad),
            "wilcoxon_test_tta_gt_zero": {
                "statistic": float(w_aurg),
                "p_value": float(p_aurg),
                "is_significant_p_lt_0001": bool(p_aurg < 0.001)
            }
        },
        "pct_mae_reduction_at_20pct_removal": stats_dict(pct_reds_20),
        "paired_comparisons_tta_vs_gradient": {
            "mean_difference": round(float(np.mean(rhos_comp) - np.mean(rhos_grad)), 6),
            "paired_t_statistic": float(t_rho_grad),
            "paired_t_p_value": float(p_rho_grad_t),
            "wilcoxon_statistic": float(w_rho_grad),
            "wilcoxon_p_value": float(p_rho_grad_w)
        }
    }

    summary_json_path = EXP014_DIR / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to {summary_json_path}")

    print("\n" + "=" * 80)
    print("EXP-014 DEVELOPMENT EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Total tiles evaluated: {len(dev_tiles)}")
    print(f"Mean Spearman rho (compensated): {summary['spearman_rho']['tta_compensated']['mean']:.4f} "
          f"[95% CI {summary['spearman_rho']['tta_compensated']['ci95']}]")
    print(f"Mean Spearman rho (uncompensated): {summary['spearman_rho']['tta_uncompensated']['mean']:.4f}")
    print(f"Mean Spearman rho (gradient baseline): {summary['spearman_rho']['gradient_baseline']['mean']:.4f}")
    print(f"Mean AURG: {summary['area_under_random_gain_aurg']['tta_aurg']['mean']:.6f} "
          f"(Wilcoxon p = {summary['area_under_random_gain_aurg']['wilcoxon_test_tta_gt_zero']['p_value']:.2e})")
    print(f"Mean MAE Reduction at 20% Sparsification: {summary['pct_mae_reduction_at_20pct_removal']['mean']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
