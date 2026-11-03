# EXP-001: Empirical Quantification of Dense Stride-1 vs. Official Stride-2 cPSNR Search

- **Experiment Date**: 2026-10-22
- **Dataset**: Exact 256 frozen validation tile predictions (`results/satlas_sr_full_validation/`)
- **Reference Ground Truth**: Exact 256 NAIP images (`data/small_val_set/naip/`)
- **Evaluator Script**: `scripts/experiment_001_dense_cpsnr.py`
- **Execution Time**: 16.29 seconds

## 1. Executive Summary & Core Results

This experiment empirically tests the hypothesis that the official PROBA-V cPSNR implementation in Satlas-SR (`ssr/metrics/cpsnr.py`), which evaluates only even-pixel offsets (`stride=2`, 81 candidate offsets), systematically underestimates alignment compared to a dense integer search (`stride=1`, 289 candidate offsets).

### Quantitative Findings
- **Mean Dense Gain**: **+0.1232 dB** (Std: 0.1848 dB)
- **95% Bootstrap Confidence Interval**: **[0.1018, 0.1477] dB** (2,000 resamples, seed 42)
- **Median Dense Gain**: **+0.0464 dB** (IQR: [0.0000, 0.1583] dB)
- **Range**: [0.0000, +1.0044] dB
- **Tiles Benefiting from Dense Search**: **176 / 256 tiles (68.75%)**
- **Tiles with Odd Best Offsets**: **176 / 256 tiles (68.75%)**

## 2. Threshold Gain Distribution

| Dense cPSNR Gain Threshold | Number of Tiles | Percentage of Dataset (N=256) |
| :--- | :--- | :--- |
| $\Delta > 0.00\text{ dB}$ (Any Improvement) | 176 | **68.75%** |
| $\Delta \ge 0.10\text{ dB}$ | 93 | **36.33%** |
| $\Delta \ge 0.25\text{ dB}$ | 43 | **16.8%** |
| $\Delta \ge 0.50\text{ dB}$ | 16 | **6.25%** |

## 3. Stratified Analysis: Odd vs. Even Offset Preference

Because the 81 stride-2 candidate offsets form an exact mathematical subset of the 289 dense candidate offsets, tiles whose true optimal displacement is even achieve $\Delta = 0.0000\text{ dB}$. Tiles whose optimal displacement has an odd coordinate achieve positive gains:

- **Tiles with Odd Coordinates in Best Offset**: **176 tiles (68.75%)**
  - Mean gain in odd subset: **+0.1792 dB**
  - Median gain in odd subset: **+0.1107 dB**
- **Tiles with Strictly Even Best Offsets**: **80 tiles (31.25%)**
  - Gain: **0.0000 dB** (identical offset selected in both grids).

## 4. Correlation with Baseline Metrics

| Baseline Metric | Pearson $r$ ($p$-value) | Spearman $\rho$ ($p$-value) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Standard PSNR** | -0.0631 (3.1429e-01) | -0.0701 (2.6392e-01) | Weak negative correlation |
| **SSIM** | +0.0317 (6.1415e-01) | +0.0139 (8.2473e-01) | Weak negative correlation |
| **LPIPS** | -0.3005 (9.6594e-07) | -0.2873 (2.9643e-06) | Weak positive correlation |
| **CLIPScore** | +0.2425 (8.8700e-05) | +0.2533 (4.1314e-05) | Weak negative correlation |

## 5. Resolution of Earlier Reporting Caveats

1. **Refutation of the 0.2–0.5 dB Heuristic**: Earlier reports conjectured that odd-pixel tiles lose '0.2 to 0.5 dB'. Our empirical measurement reveals that the **overall dataset mean gain is +0.1232 dB**, and even among the odd-shifted subset, the mean gain is **+0.1792 dB** (median: **+0.1107 dB**). Only **43 tiles (16.8%)** gain $\ge 0.25\text{ dB}$, and only **16 tiles (6.25%)** gain $\ge 0.50\text{ dB}**.
2. **Metric Sensitivity vs. Model Quality**: A dense search evaluates 289 candidates instead of 81. Because the candidate sets are nested, dense cPSNR is mathematically guaranteed to be $\ge$ stride-2 cPSNR. This metric sensitivity improvement reflects a more thorough translation search, **not an improvement in the underlying model predictions**.
3. **Paper Fidelity**: The Satlas-SR paper explicitly cites PROBA-V for its cPSNR scoring. Retaining the official stride-2 evaluation as the primary baseline metric maintains faithful reproduction of the released code, while dense search provides a valuable diagnostic on discrete translation quantization.
