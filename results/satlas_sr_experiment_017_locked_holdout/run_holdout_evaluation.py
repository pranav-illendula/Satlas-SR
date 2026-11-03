import os
import sys
import time
import json
import csv
import hashlib
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

# Fixed Paths
CHECKPOINT_PATH = Path("experiments/esrgan_8S2.pth")
DATA_ROOT_S2 = Path("../data/small_val_set/sentinel2")
DATA_ROOT_NAIP = Path("../data/small_val_set/naip")
INFERENCE_MANIFEST_PATH = Path("results/satlas_sr_full_validation/inference_manifest.csv")
PRED_ROOT = Path("results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/predictions")
SPLIT_MANIFEST_PATH = Path("results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json")
PREFLIGHT_PATH = Path("results/satlas_sr_experiment_017_locked_holdout/preflight_protocol.json")
OUT_DIR = Path("results/satlas_sr_experiment_017_locked_holdout")
MAPS_DIR = OUT_DIR / "maps"

def apply_d4_forward(x: torch.Tensor, k: int, f: int) -> torch.Tensor:
    """Exact spatial symmetry transformations: horizontal flip followed by 90-deg rotations."""
    out = x
    if f == 1:
        out = torch.flip(out, dims=[-1])
    if k != 0:
        out = torch.rot90(out, k=k, dims=[-2, -1])
    return out

def apply_d4_inverse(x: torch.Tensor, k: int, f: int) -> torch.Tensor:
    """Exact spatial inverse transformations restoring original frame coordinates."""
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
    Computes exact AUSE, AUSE_random, and AURG using numerical integration over p in [0, 0.95].
    errors, uncertainties: 1D flat arrays of length N
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.95, 20)
    
    N = len(errors)
    if np.all(uncertainties == uncertainties[0]):
        mae_initial = float(np.mean(errors))
        curve_model = np.full(len(fractions), mae_initial)
    else:
        sort_idx = np.argsort(-uncertainties)
        sorted_errors = errors[sort_idx]
        curve_model = []
        for p in fractions:
            k = int(p * N)
            rem = sorted_errors[k:]
            curve_model.append(float(np.mean(rem)) if len(rem) > 0 else 0.0)
        curve_model = np.array(curve_model)

    oracle_sort_idx = np.argsort(-errors)
    oracle_errors = errors[oracle_sort_idx]
    curve_oracle = []
    for p in fractions:
        k = int(p * N)
        rem = oracle_errors[k:]
        curve_oracle.append(float(np.mean(rem)) if len(rem) > 0 else 0.0)
    curve_oracle = np.array(curve_oracle)

    mae_initial = float(np.mean(errors))
    curve_random = np.full(len(fractions), mae_initial)

    p_steps = fractions[:len(curve_model)]
    ause = float(np.trapz(curve_model - curve_oracle, p_steps))
    ause_random = float(np.trapz(curve_random - curve_oracle, p_steps))
    aurg = float(ause_random - ause)

    def pct_red(p_target):
        idx = np.argmin(np.abs(fractions - p_target))
        val = curve_model[idx]
        return float((mae_initial - val) / mae_initial * 100.0) if mae_initial > 0 else 0.0

    return {
        "ause": round(ause, 6),
        "ause_random": round(ause_random, 6),
        "aurg": round(aurg, 6),
        "mae_initial": round(mae_initial, 6),
        "pct_red_10": round(pct_red(0.10), 3),
        "pct_red_20": round(pct_red(0.20), 3),
        "pct_red_30": round(pct_red(0.30), 3)
    }

def calc_summary_stats(series: np.ndarray):
    """Computes mean, std, median, 95% t-based CI for an array."""
    n = len(series)
    m = float(np.mean(series))
    s = float(np.std(series, ddof=1))
    med = float(np.median(series))
    t_val = stats.t.ppf(0.975, df=n - 1)
    ci_hw = t_val * s / np.sqrt(n)
    return {
        "mean": round(m, 6),
        "std": round(s, 6),
        "median": round(med, 6),
        "ci95_lower": round(m - ci_hw, 6),
        "ci95_upper": round(m + ci_hw, 6)
    }

def calc_paired_comparison(a: np.ndarray, b: np.ndarray, name_a="Method_A", name_b="Method_B"):
    """Computes paired difference statistics between array a and array b (diff = a - b)."""
    diff = a - b
    n = len(diff)
    m_diff = float(np.mean(diff))
    s_diff = float(np.std(diff, ddof=1))
    med_diff = float(np.median(diff))
    t_val = stats.t.ppf(0.975, df=n - 1)
    ci_hw = t_val * s_diff / np.sqrt(n)
    t_stat, p_t = stats.ttest_rel(a, b)
    w_stat, p_w = stats.wilcoxon(a, b)
    d = m_diff / s_diff if s_diff > 0 else 0.0
    win_count = int(np.sum(diff > 0))
    tie_count = int(np.sum(diff == 0))
    loss_count = int(np.sum(diff < 0))

    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": round(m_diff, 6),
        "std_diff": round(s_diff, 6),
        "median_diff": round(med_diff, 6),
        "ci95_lower": round(m_diff - ci_hw, 6),
        "ci95_upper": round(m_diff + ci_hw, 6),
        "t_statistic": round(float(t_stat), 6),
        "t_p_value": float(p_t),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(p_w),
        "cohens_d": round(d, 4),
        "wins": win_count,
        "ties": tie_count,
        "losses": loss_count,
        "win_rate_pct": round(win_count / n * 100.0, 2)
    }

def main():
    start_time = time.time()
    print("=" * 80)
    print("EXP-017: FROZEN INDEPENDENT HOLDOUT EVALUATION OF HYBRID UNCERTAINTY")
    print("=" * 80)

    # 1. Load Preflight Protocol and verify frozen parameters
    with open(PREFLIGHT_PATH, "r", encoding="utf-8") as f:
        preflight = json.load(f)
    assert preflight["preflight_status"] == "PASSED_FROZEN", "Preflight status is not PASSED_FROZEN!"

    holdout_tiles = preflight["holdout_tiles"]
    assert len(holdout_tiles) == 107, f"Expected 107 holdout tiles, got {len(holdout_tiles)}"

    frozen_params = preflight["frozen_hybrid_estimator"]
    alpha = float(frozen_params["alpha"])
    one_minus_alpha = float(frozen_params["one_minus_alpha"])
    z_params = frozen_params["z_score_parameters"]
    mu_tta = float(z_params["mu_tta"])
    std_tta = float(z_params["std_tta"])
    mu_input = float(z_params["mu_input"])
    std_input = float(z_params["std_input"])

    print(f"Loaded Preflight Protocol: {len(holdout_tiles)} holdout tiles verified.")
    print(f"Frozen Hybrid Parameters:")
    print(f"  alpha = {alpha:.4f}, (1 - alpha) = {one_minus_alpha:.4f}")
    print(f"  TTA Normalization:   mu = {mu_tta:.6f}, std = {std_tta:.6f}")
    print(f"  Input Normalization: mu = {mu_input:.6f}, std = {std_input:.6f}")

    # Verify zero overlap with development or quarantine sets
    with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    dev_set = set(split_data["development_tiles"])
    quar_set = set(split_data["quarantined_tiles"])
    holdout_set = set(holdout_tiles)
    assert len(holdout_set.intersection(dev_set)) == 0, "Holdout intersects development set!"
    assert len(holdout_set.intersection(quar_set)) == 0, "Holdout intersects quarantined set!"
    print("Zero overlap confirmed between holdout, dev, and quarantine partitions.")

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)

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
    print("Model checkpoint loaded successfully with strict=True.")

    # 3. Load inference manifest for S0 frame indices and NAIP paths
    with open(INFERENCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    tile_indices_map = {r["tile_id"]: [int(x) for x in r["selected_frame_indices_zero_based"].split()] for r in manifest_rows}
    tile_naip_path_map = {r["tile_id"]: Path(r["naip_reference_path"]) for r in manifest_rows}

    d4_transforms = [(k, f) for f in [0, 1] for k in [0, 1, 2, 3]]
    input_strategies = ["S0", "S1", "S3", "S4", "S5", "S6"]
    border = 4

    per_tile_rows = []
    print(f"\nBeginning single locked evaluation across {len(holdout_tiles)} holdout tiles...")

    for idx, tile in enumerate(holdout_tiles, start=1):
        tile_t0 = time.time()

        # A. Sentinel-2 TCI loading and input tensor preparation
        tci_path = DATA_ROOT_S2 / tile / "tci.png"
        tci_img = skimage.io.imread(str(tci_path))
        chunks = np.reshape(tci_img, (-1, 32, 32, 3))
        indices = tile_indices_map[tile]
        chosen_chunks = [chunks[i] for i in indices]
        tensor_chunks = [torch.as_tensor(c).permute(2, 0, 1) for c in chosen_chunks]
        input_tensor = torch.cat(tensor_chunks).unsqueeze(0).to(device).float() / 255.0

        # B. D4 TTA Inference: 8 passes
        transformed_preds = []
        for (k, f) in d4_transforms:
            x_fwd = apply_d4_forward(input_tensor, k, f)
            with torch.inference_mode():
                y_fwd = model(x_fwd)
            y_inv = apply_d4_inverse(y_fwd, k, f)
            y_inv_np = torch.clamp(y_inv, 0, 1).squeeze().cpu().numpy() # (3, 128, 128)
            transformed_preds.append(y_inv_np)

        stack_tta = np.stack(transformed_preds, axis=0) # (8, 3, 128, 128)
        base_pred_hwc = np.transpose(stack_tta[0], (1, 2, 0)) # (128, 128, 3)

        # Compute U_tta (pixel standard deviation across 8 D4 passes)
        pixel_var_c = np.var(stack_tta, axis=0, ddof=1) # (3, 128, 128)
        pixel_var = np.mean(pixel_var_c, axis=0) # (128, 128)
        pixel_std = np.sqrt(pixel_var) # (128, 128)
        u_tta_crop = pixel_std[border:128-border, border:128-border] # (120, 120)

        # C. Load EXP-011 Input Sensitivity predictions (S0..S6)
        preds_input = []
        for s in input_strategies:
            p_img = skimage.io.imread(str(PRED_ROOT / s / tile / "sr.png")).astype(np.float32) / 255.0
            preds_input.append(p_img)
        stack_input = np.stack(preds_input, axis=0) # (6, 128, 128, 3)
        var_input_c = np.var(stack_input, axis=0, ddof=1) # (128, 128, 3)
        u_input_full = np.sqrt(np.mean(var_input_c, axis=2)) # (128, 128)
        u_input_crop = u_input_full[border:128-border, border:128-border] # (120, 120)

        # Verify that base prediction matches S0 prediction from EXP-011 within rounding tolerance
        s0_pred = preds_input[0]
        max_diff_s0 = np.max(np.abs((base_pred_hwc * 255.0).round() - (s0_pred * 255.0).round()))
        assert max_diff_s0 <= 1.0, f"Tile {tile}: Base prediction differs from S0 disk prediction by {max_diff_s0}!"

        # D. Baseline 1: Sobel gradient magnitude on base prediction
        u_grad_full = compute_sobel_gradient(base_pred_hwc) # (128, 128)
        u_grad_crop = u_grad_full[border:128-border, border:128-border] # (120, 120)

        # E. Baseline 2 & 3: Constant uncertainty and Random baseline
        u_const_crop = np.full((120, 120), 0.5, dtype=np.float32)
        np.random.seed(1000 + idx)
        u_rand_crop = np.random.uniform(0.0, 1.0, size=(120, 120)).astype(np.float32)

        # F. Frozen Hybrid Uncertainty Computation
        z_tta = (u_tta_crop - mu_tta) / (std_tta + 1e-8)
        z_input = (u_input_crop - mu_input) / (std_input + 1e-8)
        u_hybrid_crop = alpha * z_tta + one_minus_alpha * z_input # (120, 120)

        # G. NAIP Reference loading, Registration, and Error Computation
        naip_path = tile_naip_path_map[tile]
        naip_img = skimage.io.imread(str(naip_path)).astype(np.float32) / 255.0 # (128, 128, 3)

        best_dx, best_dy, best_mse = compute_registration_shift(base_pred_hwc, naip_img, max_shift=8, border=border)
        shift_mag = float(np.sqrt(best_dx ** 2 + best_dy ** 2))

        # Compensated error crop
        cH, cW = 128 - 2 * border, 128 - 2 * border
        y_ref_start = border + best_dy
        x_ref_start = border + best_dx
        ref_comp_crop = naip_img[y_ref_start:y_ref_start+cH, x_ref_start:x_ref_start+cW]
        base_pred_crop = base_pred_hwc[border:128-border, border:128-border]
        err_comp_crop = np.mean(np.abs(base_pred_crop - ref_comp_crop), axis=2) # (120, 120)

        # Uncompensated error crop
        err_uncomp_full = np.mean(np.abs(base_pred_hwc - naip_img), axis=2) # (128, 128)
        err_uncomp_crop = err_uncomp_full[border:128-border, border:128-border] # (120, 120)

        # H. Flatten interior arrays for statistical error-ranking
        f_err_comp = err_comp_crop.flatten()
        f_err_uncomp = err_uncomp_crop.flatten()

        f_hyb = u_hybrid_crop.flatten()
        f_input = u_input_crop.flatten()
        f_tta = u_tta_crop.flatten()
        f_grad = u_grad_crop.flatten()
        f_rand = u_rand_crop.flatten()
        f_const = u_const_crop.flatten()

        # Spearman rank correlations vs compensated error
        rho_hyb, p_hyb = stats.spearmanr(f_hyb, f_err_comp)
        rho_input, p_input = stats.spearmanr(f_input, f_err_comp)
        rho_tta, p_tta = stats.spearmanr(f_tta, f_err_comp)
        rho_grad, p_grad = stats.spearmanr(f_grad, f_err_comp)
        rho_rand, p_rand = stats.spearmanr(f_rand, f_err_comp)
        rho_const = 0.0

        # Spearman rank correlations vs uncompensated error
        rho_hyb_u, _ = stats.spearmanr(f_hyb, f_err_uncomp)
        rho_input_u, _ = stats.spearmanr(f_input, f_err_uncomp)
        rho_tta_u, _ = stats.spearmanr(f_tta, f_err_uncomp)
        rho_grad_u, _ = stats.spearmanr(f_grad, f_err_uncomp)

        # Sparsification metrics on compensated error
        sp_hyb = compute_sparsification_metrics(f_err_comp, f_hyb)
        sp_input = compute_sparsification_metrics(f_err_comp, f_input)
        sp_tta = compute_sparsification_metrics(f_err_comp, f_tta)
        sp_grad = compute_sparsification_metrics(f_err_comp, f_grad)
        sp_rand = compute_sparsification_metrics(f_err_comp, f_rand)
        sp_const = compute_sparsification_metrics(f_err_comp, f_const)

        # Save per-tile map artifact
        map_path = MAPS_DIR / f"{tile}_holdout_maps.npz"
        np.savez_compressed(
            str(map_path),
            u_hybrid=u_hybrid_crop.astype(np.float32),
            u_tta=pixel_std.astype(np.float32),
            u_input=u_input_full.astype(np.float32),
            u_grad=u_grad_full.astype(np.float32),
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
            "mae_comp_initial": sp_hyb["mae_initial"],
            "mae_uncomp": round(float(np.mean(f_err_uncomp)), 6),
            # Compensated Spearman rho
            "rho_hybrid_comp": round(float(rho_hyb), 6),
            "p_hybrid_comp": float(p_hyb),
            "rho_input_comp": round(float(rho_input), 6),
            "p_input_comp": float(p_input),
            "rho_tta_comp": round(float(rho_tta), 6),
            "p_tta_comp": float(p_tta),
            "rho_grad_comp": round(float(rho_grad), 6),
            "p_grad_comp": float(p_grad),
            "rho_rand_comp": round(float(rho_rand), 6),
            "rho_const_comp": 0.0,
            # Uncompensated Spearman rho
            "rho_hybrid_uncomp": round(float(rho_hyb_u), 6),
            "rho_input_uncomp": round(float(rho_input_u), 6),
            "rho_tta_uncomp": round(float(rho_tta_u), 6),
            "rho_grad_uncomp": round(float(rho_grad_u), 6),
            # Paired difference vs Sobel gradient (Primary metric)
            "delta_rho_hyb_minus_grad": round(float(rho_hyb - rho_grad), 6),
            "delta_rho_input_minus_grad": round(float(rho_input - rho_grad), 6),
            "delta_rho_tta_minus_grad": round(float(rho_tta - rho_grad), 6),
            "delta_rho_hyb_minus_input": round(float(rho_hyb - rho_input), 6),
            # AURG
            "aurg_hybrid": sp_hyb["aurg"],
            "aurg_input": sp_input["aurg"],
            "aurg_tta": sp_tta["aurg"],
            "aurg_grad": sp_grad["aurg"],
            "aurg_rand": sp_rand["aurg"],
            "aurg_const": sp_const["aurg"],
            # AUSE
            "ause_hybrid": sp_hyb["ause"],
            "ause_input": sp_input["ause"],
            "ause_tta": sp_tta["ause"],
            "ause_grad": sp_grad["ause"],
            "ause_rand": sp_rand["ause"],
            "ause_const": sp_const["ause"],
            # MAE Reduction at 10%
            "pct_red_10_hybrid": sp_hyb["pct_red_10"],
            "pct_red_10_input": sp_input["pct_red_10"],
            "pct_red_10_tta": sp_tta["pct_red_10"],
            "pct_red_10_grad": sp_grad["pct_red_10"],
            # MAE Reduction at 20%
            "pct_red_20_hybrid": sp_hyb["pct_red_20"],
            "pct_red_20_input": sp_input["pct_red_20"],
            "pct_red_20_tta": sp_tta["pct_red_20"],
            "pct_red_20_grad": sp_grad["pct_red_20"],
            # MAE Reduction at 30%
            "pct_red_30_hybrid": sp_hyb["pct_red_30"],
            "pct_red_30_input": sp_input["pct_red_30"],
            "pct_red_30_tta": sp_tta["pct_red_30"],
            "pct_red_30_grad": sp_grad["pct_red_30"],
        }
        per_tile_rows.append(record)

        elapsed_tile = time.time() - tile_t0
        if idx % 10 == 0 or idx == len(holdout_tiles):
            print(f"[{idx:3d}/{len(holdout_tiles)}] Tile {tile} processed in {elapsed_tile:.2f}s | "
                  f"Rho Hyb: {rho_hyb:.4f}, Sobel: {rho_grad:.4f}, Delta: {rho_hyb - rho_grad:+.4f}")

    total_eval_time = time.time() - start_time
    print(f"\nCompleted evaluation of {len(per_tile_rows)} holdout tiles in {total_eval_time:.2f}s "
          f"({total_eval_time / len(per_tile_rows):.2f}s/tile).")

    # 4. Save per-tile CSV
    csv_path = OUT_DIR / "per_tile_holdout_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_tile_rows)
    print(f"Saved per-tile CSV: {csv_path}")

    # 5. Statistical Aggregations
    arr_hyb = np.array([r["rho_hybrid_comp"] for r in per_tile_rows])
    arr_input = np.array([r["rho_input_comp"] for r in per_tile_rows])
    arr_tta = np.array([r["rho_tta_comp"] for r in per_tile_rows])
    arr_grad = np.array([r["rho_grad_comp"] for r in per_tile_rows])
    arr_rand = np.array([r["rho_rand_comp"] for r in per_tile_rows])
    arr_const = np.array([r["rho_const_comp"] for r in per_tile_rows])

    arr_hyb_u = np.array([r["rho_hybrid_uncomp"] for r in per_tile_rows])
    arr_input_u = np.array([r["rho_input_uncomp"] for r in per_tile_rows])
    arr_tta_u = np.array([r["rho_tta_uncomp"] for r in per_tile_rows])
    arr_grad_u = np.array([r["rho_grad_uncomp"] for r in per_tile_rows])

    arr_aurg_hyb = np.array([r["aurg_hybrid"] for r in per_tile_rows])
    arr_aurg_input = np.array([r["aurg_input"] for r in per_tile_rows])
    arr_aurg_tta = np.array([r["aurg_tta"] for r in per_tile_rows])
    arr_aurg_grad = np.array([r["aurg_grad"] for r in per_tile_rows])

    arr_p20_hyb = np.array([r["pct_red_20_hybrid"] for r in per_tile_rows])
    arr_p20_input = np.array([r["pct_red_20_input"] for r in per_tile_rows])
    arr_p20_tta = np.array([r["pct_red_20_tta"] for r in per_tile_rows])
    arr_p20_grad = np.array([r["pct_red_20_grad"] for r in per_tile_rows])

    arr_shifts = np.array([r["reg_shift_mag_px"] for r in per_tile_rows])

    # Paired comparisons
    comp_hyb_vs_sobel = calc_paired_comparison(arr_hyb, arr_grad, "Hybrid", "Sobel_Gradient")
    comp_hyb_vs_input = calc_paired_comparison(arr_hyb, arr_input, "Hybrid", "Input_Sensitivity")
    comp_hyb_vs_tta = calc_paired_comparison(arr_hyb, arr_tta, "Hybrid", "D4_TTA")
    comp_input_vs_sobel = calc_paired_comparison(arr_input, arr_grad, "Input_Sensitivity", "Sobel_Gradient")
    comp_tta_vs_sobel = calc_paired_comparison(arr_tta, arr_grad, "D4_TTA", "Sobel_Gradient")

    # Secondary paired comparisons for AURG and 20% reduction
    comp_aurg_hyb_vs_sobel = calc_paired_comparison(arr_aurg_hyb, arr_aurg_grad, "AURG_Hybrid", "AURG_Sobel")
    comp_p20_hyb_vs_sobel = calc_paired_comparison(arr_p20_hyb, arr_p20_grad, "PctRed20_Hybrid", "PctRed20_Sobel")

    summary = {
        "experiment_id": "EXP-017",
        "title": "Frozen Independent Holdout Evaluation of Hybrid Uncertainty",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_holdout_tiles": len(per_tile_rows),
        "total_runtime_seconds": round(total_eval_time, 2),
        "seconds_per_tile": round(total_eval_time / len(per_tile_rows), 2),
        "frozen_parameters": {
            "alpha": alpha,
            "one_minus_alpha": one_minus_alpha,
            "mu_tta": mu_tta,
            "std_tta": std_tta,
            "mu_input": mu_input,
            "std_input": std_input
        },
        "registration_statistics": {
            "mean_shift_px": round(float(np.mean(arr_shifts)), 3),
            "median_shift_px": round(float(np.median(arr_shifts)), 3),
            "max_shift_px": round(float(np.max(arr_shifts)), 3),
            "pct_nonzero_shift": round(float(np.mean(arr_shifts > 0) * 100.0), 2)
        },
        "spearman_rho_compensated": {
            "hybrid": calc_summary_stats(arr_hyb),
            "input_sensitivity": calc_summary_stats(arr_input),
            "d4_tta": calc_summary_stats(arr_tta),
            "sobel_gradient": calc_summary_stats(arr_grad),
            "random_baseline": calc_summary_stats(arr_rand),
            "constant_baseline": calc_summary_stats(arr_const)
        },
        "spearman_rho_uncompensated": {
            "hybrid": calc_summary_stats(arr_hyb_u),
            "input_sensitivity": calc_summary_stats(arr_input_u),
            "d4_tta": calc_summary_stats(arr_tta_u),
            "sobel_gradient": calc_summary_stats(arr_grad_u)
        },
        "primary_endpoint_paired_differences": {
            "hybrid_vs_sobel": comp_hyb_vs_sobel,
            "input_vs_sobel": comp_input_vs_sobel,
            "tta_vs_sobel": comp_tta_vs_sobel,
            "hybrid_vs_input": comp_hyb_vs_input,
            "hybrid_vs_tta": comp_hyb_vs_tta
        },
        "secondary_endpoints": {
            "aurg": {
                "hybrid": calc_summary_stats(arr_aurg_hyb),
                "input_sensitivity": calc_summary_stats(arr_aurg_input),
                "d4_tta": calc_summary_stats(arr_aurg_tta),
                "sobel_gradient": calc_summary_stats(arr_aurg_grad),
                "paired_hybrid_vs_sobel": comp_aurg_hyb_vs_sobel
            },
            "pct_reduction_10": {
                "hybrid": calc_summary_stats(np.array([r["pct_red_10_hybrid"] for r in per_tile_rows])),
                "sobel_gradient": calc_summary_stats(np.array([r["pct_red_10_grad"] for r in per_tile_rows]))
            },
            "pct_reduction_20": {
                "hybrid": calc_summary_stats(arr_p20_hyb),
                "input_sensitivity": calc_summary_stats(arr_p20_input),
                "d4_tta": calc_summary_stats(arr_p20_tta),
                "sobel_gradient": calc_summary_stats(arr_p20_grad),
                "paired_hybrid_vs_sobel": comp_p20_hyb_vs_sobel
            },
            "pct_reduction_30": {
                "hybrid": calc_summary_stats(np.array([r["pct_red_30_hybrid"] for r in per_tile_rows])),
                "sobel_gradient": calc_summary_stats(np.array([r["pct_red_30_grad"] for r in per_tile_rows]))
            }
        },
        "development_vs_holdout_replication": {
            "development_exp015_cv": {
                "mean_rho_hybrid": 0.30476,
                "mean_rho_sobel": 0.22241,
                "paired_mean_delta": 0.08235,
                "relative_gain_pct": 37.03,
                "t_p_value": 3.20e-11,
                "win_rate_pct": 77.57
            },
            "holdout_exp017": {
                "mean_rho_hybrid": comp_hyb_vs_sobel["mean_diff"] + calc_summary_stats(arr_grad)["mean"],
                "mean_rho_sobel": calc_summary_stats(arr_grad)["mean"],
                "paired_mean_delta": comp_hyb_vs_sobel["mean_diff"],
                "relative_gain_pct": round((comp_hyb_vs_sobel["mean_diff"] / calc_summary_stats(arr_grad)["mean"]) * 100.0, 2),
                "t_p_value": comp_hyb_vs_sobel["t_p_value"],
                "win_rate_pct": comp_hyb_vs_sobel["win_rate_pct"]
            }
        }
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    print("\n" + "=" * 80)
    print("EXP-017 HOLDOUT EVALUATION COMPLETE")
    print(f"Primary Endpoint (Hybrid vs Sobel Paired Delta Spearman Rho):")
    print(f"  Hybrid Mean Rho: {calc_summary_stats(arr_hyb)['mean']:.4f} (95% CI: [{calc_summary_stats(arr_hyb)['ci95_lower']:.4f}, {calc_summary_stats(arr_hyb)['ci95_upper']:.4f}])")
    print(f"  Sobel  Mean Rho: {calc_summary_stats(arr_grad)['mean']:.4f} (95% CI: [{calc_summary_stats(arr_grad)['ci95_lower']:.4f}, {calc_summary_stats(arr_grad)['ci95_upper']:.4f}])")
    print(f"  Paired Delta:    {comp_hyb_vs_sobel['mean_diff']:+.4f} (95% CI: [{comp_hyb_vs_sobel['ci95_lower']:+.4f}, {comp_hyb_vs_sobel['ci95_upper']:+.4f}])")
    print(f"  t-statistic:     {comp_hyb_vs_sobel['t_statistic']:.4f} (p = {comp_hyb_vs_sobel['t_p_value']:.4e})")
    print(f"  Wilcoxon test:   W = {comp_hyb_vs_sobel['wilcoxon_stat']:.1f} (p = {comp_hyb_vs_sobel['wilcoxon_p_value']:.4e})")
    print(f"  Cohen's d:       {comp_hyb_vs_sobel['cohens_d']:.4f}")
    print(f"  Win Rate:        {comp_hyb_vs_sobel['win_rate_pct']:.2f}% ({comp_hyb_vs_sobel['wins']}/{len(per_tile_rows)})")
    print("=" * 80)

if __name__ == "__main__":
    main()
