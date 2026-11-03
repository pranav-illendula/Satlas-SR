import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import scipy.stats as stats

HOLDOUT_CSV = Path("results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv")
HOLDOUT_SUMMARY = Path("results/satlas_sr_experiment_017_locked_holdout/summary.json")

with open(HOLDOUT_CSV, "r", encoding="utf-8") as f:
    holdout_rows = list(csv.DictReader(f))

with open(HOLDOUT_SUMMARY, "r", encoding="utf-8") as f:
    holdout_summary = json.load(f)

print(f"Total holdout rows: {len(holdout_rows)}")

# Helper to compute summary stats
def calc_stats(arr):
    n = len(arr)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    med = float(np.median(arr))
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_hw = t_crit * s / np.sqrt(n)
    return {
        "mean": m,
        "std": s,
        "median": med,
        "ci95_lower": m - ci_hw,
        "ci95_upper": m + ci_hw
    }

def calc_paired(a, b):
    diff = a - b
    n = len(diff)
    m = float(np.mean(diff))
    s = float(np.std(diff, ddof=1))
    med = float(np.median(diff))
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_hw = t_crit * s / np.sqrt(n)
    t_stat, p_t = stats.ttest_rel(a, b)
    w_stat, p_w = stats.wilcoxon(a, b)
    d = m / s if s > 0 else 0.0
    wins = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    losses = int(np.sum(diff < 0))
    return {
        "mean_diff": m,
        "std_diff": s,
        "median_diff": med,
        "ci95_lower": m - ci_hw,
        "ci95_upper": m + ci_hw,
        "t_stat": float(t_stat),
        "t_p_val": float(p_t),
        "w_stat": float(w_stat),
        "w_p_val": float(p_w),
        "cohens_d": d,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / n * 100.0
    }

# Arrays
rho_hyb = np.array([float(r["rho_hybrid_comp"]) for r in holdout_rows])
rho_sobel = np.array([float(r["rho_grad_comp"]) for r in holdout_rows])
rho_input = np.array([float(r["rho_input_comp"]) for r in holdout_rows])
rho_tta = np.array([float(r["rho_tta_comp"]) for r in holdout_rows])
rho_rand = np.array([float(r["rho_rand_comp"]) for r in holdout_rows])
rho_const = np.array([float(r["rho_const_comp"]) for r in holdout_rows])

rho_hyb_u = np.array([float(r["rho_hybrid_uncomp"]) for r in holdout_rows])
rho_sobel_u = np.array([float(r["rho_grad_uncomp"]) for r in holdout_rows])
rho_input_u = np.array([float(r["rho_input_uncomp"]) for r in holdout_rows])
rho_tta_u = np.array([float(r["rho_tta_uncomp"]) for r in holdout_rows])

aurg_hyb = np.array([float(r["aurg_hybrid"]) for r in holdout_rows])
aurg_sobel = np.array([float(r["aurg_grad"]) for r in holdout_rows])
aurg_input = np.array([float(r["aurg_input"]) for r in holdout_rows])
aurg_tta = np.array([float(r["aurg_tta"]) for r in holdout_rows])

p10_hyb = np.array([float(r["pct_red_10_hybrid"]) for r in holdout_rows])
p10_sobel = np.array([float(r["pct_red_10_grad"]) for r in holdout_rows])
p20_hyb = np.array([float(r["pct_red_20_hybrid"]) for r in holdout_rows])
p20_sobel = np.array([float(r["pct_red_20_grad"]) for r in holdout_rows])
p20_input = np.array([float(r["pct_red_20_input"]) for r in holdout_rows])
p20_tta = np.array([float(r["pct_red_20_tta"]) for r in holdout_rows])
p30_hyb = np.array([float(r["pct_red_30_hybrid"]) for r in holdout_rows])
p30_sobel = np.array([float(r["pct_red_30_grad"]) for r in holdout_rows])

# Verify Primary Endpoint: Hybrid vs Sobel
p_hyb_vs_sobel = calc_paired(rho_hyb, rho_sobel)
print("\n--- Primary Endpoint: Recomputed vs Reported ---")
rep_prim = holdout_summary["primary_endpoint_paired_differences"]["hybrid_vs_sobel"]
for k in ["mean_diff", "std_diff", "median_diff", "ci95_lower", "ci95_upper", "t_statistic", "t_p_value", "wilcoxon_stat", "wilcoxon_p_value", "cohens_d", "wins", "ties", "losses", "win_rate_pct"]:
    rep_k = rep_prim.get(k)
    # map to recomputed key
    map_k = {"t_statistic": "t_stat", "t_p_value": "t_p_val", "wilcoxon_stat": "w_stat", "wilcoxon_p_value": "w_p_val", "win_rate_pct": "win_rate"}.get(k, k)
    rec_val = p_hyb_vs_sobel[map_k]
    diff_val = abs(rec_val - rep_k) if isinstance(rec_val, (int, float)) else None
    print(f"  {k:18s}: Recomputed={rec_val:12.6f} | Reported={rep_k:12.6f} | Match={diff_val < 1e-4}")

print("\n--- Secondary Endpoint: AURG Recomputed vs Reported ---")
p_aurg_hyb_vs_sobel = calc_paired(aurg_hyb, aurg_sobel)
rep_aurg = holdout_summary["secondary_endpoints"]["aurg"]["paired_hybrid_vs_sobel"]
for k in ["mean_diff", "std_diff", "t_statistic", "t_p_value", "wilcoxon_stat", "wilcoxon_p_value", "cohens_d", "wins", "losses"]:
    rep_k = rep_aurg.get(k)
    map_k = {"t_statistic": "t_stat", "t_p_value": "t_p_val", "wilcoxon_stat": "w_stat", "wilcoxon_p_value": "w_p_val"}.get(k, k)
    rec_val = p_aurg_hyb_vs_sobel[map_k]
    diff_val = abs(rec_val - rep_k) if isinstance(rec_val, (int, float)) else None
    print(f"  {k:18s}: Recomputed={rec_val:12.6f} | Reported={rep_k:12.6f} | Match={diff_val < 1e-4}")

print("\n--- Secondary Endpoint: 20% Sparsification MAE Reduction Recomputed vs Reported ---")
p_p20_hyb_vs_sobel = calc_paired(p20_hyb, p20_sobel)
rep_p20 = holdout_summary["secondary_endpoints"]["pct_reduction_20"]["paired_hybrid_vs_sobel"]
for k in ["mean_diff", "std_diff", "t_statistic", "t_p_value", "wilcoxon_stat", "wilcoxon_p_value", "cohens_d", "wins", "losses"]:
    rep_k = rep_p20.get(k)
    map_k = {"t_statistic": "t_stat", "t_p_value": "t_p_val", "wilcoxon_stat": "w_stat", "wilcoxon_p_value": "w_p_val"}.get(k, k)
    rec_val = p_p20_hyb_vs_sobel[map_k]
    diff_val = abs(rec_val - rep_k) if isinstance(rec_val, (int, float)) else None
    print(f"  {k:18s}: Recomputed={rec_val:12.6f} | Reported={rep_k:12.6f} | Match={diff_val < 1e-4}")
