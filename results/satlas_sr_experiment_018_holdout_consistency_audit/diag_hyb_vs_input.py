import json
import csv
from pathlib import Path
import numpy as np
import scipy.stats as stats

EXP017_CSV = Path("results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv")
with open(EXP017_CSV, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total holdout rows: {len(rows)}")

hyb = np.array([float(r["rho_hybrid_comp"]) for r in rows])
inp = np.array([float(r["rho_input_comp"]) for r in rows])
diff = hyb - inp

print("\n--- Hybrid vs Input Sensitivity on Holdout ---")
print(f"Mean diff: {np.mean(diff):.6f}")
print(f"Std diff: {np.std(diff, ddof=1):.6f}")
print(f"Median diff: {np.median(diff):.6f}")
print(f"Min diff: {np.min(diff):.6f}, Max diff: {np.max(diff):.6f}")
print(f"IQR: {stats.iqr(diff):.6f}")
print(f"Skewness: {stats.skew(diff):.4f}")
print(f"Kurtosis: {stats.kurtosis(diff):.4f}")

# Paired t-test
t_stat, p_t = stats.ttest_rel(hyb, inp)
print(f"Paired t-test: t = {t_stat:.6f}, p = {p_t:.6f}")

# Wilcoxon signed-rank test
w_stat, p_w = stats.wilcoxon(hyb, inp)
print(f"Wilcoxon signed-rank: W = {w_stat}, p = {p_w:.8f}")

# Wins, losses, ties
wins = np.sum(diff > 0)
losses = np.sum(diff < 0)
ties = np.sum(diff == 0)
print(f"Wins: {wins}, Losses: {losses}, Ties: {ties} (Win rate: {wins/len(diff)*100:.2f}%)")

# Inspect negative outliers
neg_outliers = diff[diff < -0.05]
print(f"Negative outliers (< -0.05): {neg_outliers}")
pos_outliers = diff[diff > 0.05]
print(f"Positive outliers (> 0.05): {pos_outliers}")

# Quartiles of diff
print(f"25th percentile: {np.percentile(diff, 25):.6f}")
print(f"50th percentile (median): {np.percentile(diff, 50):.6f}")
print(f"75th percentile: {np.percentile(diff, 75):.6f}")
print(f"Positive values count: {np.sum(diff > 0)} / {len(diff)}")
