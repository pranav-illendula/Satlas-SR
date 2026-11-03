import os
import sys
import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import scipy.stats as stats

# Paths
REPO_ROOT = Path(".").resolve()
EXP015_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_015_hybrid_uncertainty"
EXP016_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_016_independent_audit"
EXP017_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_017_locked_holdout"
OUT_DIR = REPO_ROOT / "results" / "satlas_sr_experiment_018_holdout_consistency_audit"

print("--- Step 1: Inspecting Development CSV (EXP-015 per_tile_oof_metrics.csv) ---")
dev_csv = EXP015_DIR / "per_tile_oof_metrics.csv"
with open(dev_csv, "r", encoding="utf-8") as f:
    dev_rows = list(csv.DictReader(f))
print(f"Total dev rows: {len(dev_rows)}")
print("Columns in dev CSV:", list(dev_rows[0].keys()))

# Check Hybrid vs Sobel on dev
dev_hyb = np.array([float(r["rho_hybrid_comp"]) for r in dev_rows])
dev_sobel = np.array([float(r["rho_grad_comp"]) for r in dev_rows])
dev_diff = dev_hyb - dev_sobel
dev_wins_hyb_vs_sobel = np.sum(dev_diff > 0)
dev_losses_hyb_vs_sobel = np.sum(dev_diff < 0)
dev_ties_hyb_vs_sobel = np.sum(dev_diff == 0)
print(f"Dev Hybrid vs Sobel (compensated): wins={dev_wins_hyb_vs_sobel}, losses={dev_losses_hyb_vs_sobel}, ties={dev_ties_hyb_vs_sobel}")
print(f"Dev Hybrid mean: {np.mean(dev_hyb):.6f}, Sobel mean: {np.mean(dev_sobel):.6f}, diff mean: {np.mean(dev_diff):.6f}")

# Check what yields 83 wins and 24 losses!
# Could it be uncompensated error? Or win count vs something else?
for col_a in ["rho_hybrid_comp", "rho_hybrid_uncomp"]:
    for col_b in ["rho_grad_comp", "rho_grad_uncomp", "rho_tta_comp", "rho_input_comp"]:
        arr_a = np.array([float(r[col_a]) for r in dev_rows])
        arr_b = np.array([float(r[col_b]) for r in dev_rows])
        d = arr_a - arr_b
        w = np.sum(d > 0)
        l = np.sum(d < 0)
        print(f"  Comparison {col_a} vs {col_b}: wins={w}, losses={l}, win_rate={w/len(d)*100:.2f}%")

# Also check fold-averaged means vs tile-averaged means
folds = sorted(list(set(r["cv_fold"] for r in dev_rows)))
print("\nFold-level means:")
fold_hyb_means = []
fold_sobel_means = []
fold_diff_means = []
for f in folds:
    f_rows = [r for r in dev_rows if r["cv_fold"] == f]
    f_hyb = np.mean([float(r["rho_hybrid_comp"]) for r in f_rows])
    f_sobel = np.mean([float(r["rho_grad_comp"]) for r in f_rows])
    f_diff = np.mean([float(r["rho_hybrid_comp"]) - float(r["rho_grad_comp"]) for r in f_rows])
    fold_hyb_means.append(f_hyb)
    fold_sobel_means.append(f_sobel)
    fold_diff_means.append(f_diff)
    print(f"  Fold {f}: N={len(f_rows)}, Hyb={f_hyb:.5f}, Sobel={f_sobel:.5f}, Diff={f_diff:.5f}")
print(f"Macro-average across 5 folds: Hyb={np.mean(fold_hyb_means):.5f}, Sobel={np.mean(fold_sobel_means):.5f}, Diff={np.mean(fold_diff_means):.5f}")

# Check where the numbers 0.30476, 0.22241, 0.08235, 77.57% (83/107) came from!
print("\n--- Searching for where 0.30476, 0.22241, 0.08235, 77.57% came from in other files ---")
