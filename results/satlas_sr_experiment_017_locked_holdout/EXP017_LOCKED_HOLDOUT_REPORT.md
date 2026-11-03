# EXP-017: Frozen Independent Holdout Evaluation of Hybrid Uncertainty

**Status:** COMPLETE (FROZEN)  
**Date:** 2026-10-23  
**Repository:** `satlas-super-resolution`  
**Git Commit Hash:** `3e67973f6746bedbd91cc032b4888574980e1daf`  
**Execution Environment:** Windows (x64), NVIDIA RTX 2000 Ada Generation, Python 3.10 venv  
**Directory:** `results/satlas_sr_experiment_017_locked_holdout/`  
**Related Experiments:** EXP-013 (Readiness Audit), EXP-014 ($D_4$ TTA Evaluation), EXP-015 (Hybrid CV), EXP-016 (Independent Audit)

---

## 1. Executive Summary

In **EXP-017**, the frozen hybrid uncertainty estimator developed in EXP-015 and audited in EXP-016 was evaluated **exactly once** on the pristine **107-tile locked holdout partition**. The holdout partition had remained strictly unopened, uninspected, and unreferenced throughout all prior experiments. All parameters—the hybrid weighting coefficient ($\alpha^* = 0.1900$), the feature standardization statistics ($\mu, \sigma$), the evaluation window (interior $120 \times 120$ pixels, excluding border 4), and the integer registration shift policy ($[\Delta x, \Delta y] \in [-8, 8] \times [-8, 8]$)—were frozen prior to accessing holdout data.

### Key Evaluation Findings

1. **Primary Endpoint Replicated with High Statistical Significance:**
   - On the 107 holdout tiles, the frozen **Hybrid Uncertainty Estimator** achieved a mean per-tile Spearman error-ranking correlation of **$\bar{\rho} = 0.2800$** (95% CI: $[0.2513, 0.3086]$), compared to **$\bar{\rho} = 0.2006$** (95% CI: $[0.1733, 0.2280]$) for the **Sobel Gradient Baseline**.
   - The primary endpoint—the paired per-tile difference $\Delta \rho = \rho_{\text{hybrid}} - \rho_{\text{grad}}$—is **$\bar{\Delta} = +0.0793$** (95% CI: $[+0.0611, +0.0975]$), representing a **$+39.53\%$ relative gain** over the baseline.
   - Paired tests confirm strong statistical superiority: paired $t = 8.6251$ ($p = 6.87 \times 10^{-14}$), Wilcoxon signed-rank $W = 458.0$ ($p = 4.17 \times 10^{-14}$).
   - The effect size is large: **Cohen's $d = 0.8338$**.
   - Hybrid uncertainty outperformed the Sobel baseline on **93 out of 107 holdout tiles (86.92% win rate)** with zero ties.

2. **Replication of Development Findings:**
   - In EXP-015 5-fold cross-validation, the hybrid estimator achieved $\bar{\rho} = 0.3048$, a paired gain over Sobel of $\bar{\Delta} = +0.0824$ ($+37.03\%$ relative gain), and a win rate of $77.57\%$.
   - On the independent holdout partition, the observed paired gain ($\bar{\Delta} = +0.0793$) matches the development effect within $0.0031$ correlation units, confirming that development-set gains were not the product of fold overfitting, threshold tuning, or exploratory data snooping.

3. **Secondary Sparsification Endpoints:**
   - **Area Under the Random Error Retention Gain Curve (AURG):** Hybrid achieved $\text{AURG} = 0.01593$ vs Sobel $\text{AURG} = 0.01124$ (paired $\bar{\Delta} = +0.00469$, $p = 2.96 \times 10^{-14}$, win rate $87.85\%$).
   - **MAE Reduction at 20% Sparsification:** Discarding the top 20% highest-uncertainty pixels reduced reconstruction MAE by **$11.27\%$** for Hybrid vs **$8.43\%$** for Sobel (paired $\bar{\Delta} = +2.84\%$, $p = 2.75 \times 10^{-16}$, Cohen's $d = 0.9371$, win rate $91.59\%$).

4. **Component Contributions Confirmed:**
   - As observed on the development partition, **Sentinel-2 Input-Selection Sensitivity** ($U_{\text{input}}$) provides the majority of the ranking power ($\bar{\Delta} = +0.0748$ over Sobel, $p = 3.13 \times 10^{-11}$).
   - Combining $D_4$ TTA orientation sensitivity with input sensitivity ($U_{\text{hybrid}} = 0.19 Z(U_{\text{tta}}) + 0.81 Z(U_{\text{input}})$) yields a consistent incremental improvement: Hybrid beats Input-Sensitivity alone on **71 out of 107 tiles (66.36% win rate)**, with a statistically significant Wilcoxon signed-rank test ($p = 0.000587$).

---

## 2. Preflight Verification and Protocol Freeze

Before opening any holdout image files or computing holdout metrics, strict preflight audit criteria were established and validated in `preflight_protocol.json`:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    PREFLIGHT VERIFICATION SUMMARY                          │
├─────────────────────────────────────────┬──────────────────────────────────┤
│ Criterion                               │ Status                           │
├─────────────────────────────────────────┼──────────────────────────────────┤
│ Holdout Partition Count                 │ Exactly 107 tiles                │
│ Overlap with Development Partition      │ 0 tiles (Verified Disjoint)      │
│ Overlap with Quarantined Partition      │ 0 tiles (Verified Disjoint)      │
│ Prior Holdout Error Exposure            │ Zero (Holdout was locked)        │
│ Tuning on Holdout Results               │ Forbidden (Strictly single pass) │
│ Frozen Parameter Derivation Source      │ 100% Development set only        │
└─────────────────────────────────────────┴──────────────────────────────────┘
```

### Frozen Parameters (Derived Pre-Holdout)
- **Hybrid Combination Weight:** $\alpha^* = 0.1900$, $(1 - \alpha^*) = 0.8100$
- **$D_4$ TTA Normalization:** $\mu_{\text{tta}} = 0.083126, \sigma_{\text{tta}} = 0.051388$
- **Input Sensitivity Normalization:** $\mu_{\text{input}} = 0.057286, \sigma_{\text{input}} = 0.041238$
- **Spatial Evaluation Window:** Central $120 \times 120$ pixels (excluding a 4-pixel border to match cPSNR registration protocols)
- **Integer Registration Shift:** Optimal integer translation $(\Delta x, \Delta y) \in [-8, 8] \times [-8, 8]$ minimizing MSE on the central crop.

---

## 3. Comprehensive Holdout Evaluation Results

All 107 holdout tiles were evaluated successfully with zero exclusions or dropped tiles.

### 3.1 Per-Method Performance on Compensated Errors

| Estimator / Baseline | Spearman $\bar{\rho}$ | Sample SD | Median $\rho$ | 95% Parametric CI | Win Rate vs Sobel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frozen Hybrid Estimator** ($U_{\text{hybrid}}$) | **0.2800** | 0.1496 | **0.2957** | **[0.2513, 0.3086]** | **86.92% (93/107)** |
| Input Sensitivity ($U_{\text{input}}$) | 0.2754 | 0.1488 | 0.2901 | [0.2469, 0.3040] | 85.05% (91/107) |
| $D_4$ TTA Uncertainty ($U_{\text{tta}}$) | 0.1700 | 0.1520 | 0.1713 | [0.1408, 0.1991] | 31.78% (34/107) |
| Sobel Gradient Baseline ($U_{\text{grad}}$) | 0.2006 | 0.1425 | 0.2115 | [0.1733, 0.2280] | Reference Baseline |
| Random Uniform Baseline ($U_{\text{rand}}$) | -0.0012 | 0.0080 | -0.0013 | [-0.0027, 0.0004] | 0.00% (0/107) |
| Constant Baseline ($U_{\text{const}}$) | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.00% (0/107) |

### 3.2 Primary Endpoint: Paired Statistical Comparisons

The primary endpoint is the paired per-tile difference against the Sobel gradient baseline.

| Paired Comparison | Mean $\bar{\Delta}$ | SD of $\Delta$ | 95% CI of $\Delta$ | Paired $t$-stat | $t$-test $p$-value | Wilcoxon $W$ | Wilcoxon $p$-value | Cohen's $d$ | Wins / Losses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid vs Sobel** (Primary) | **+0.0793** | 0.0951 | **[+0.0611, +0.0975]** | **8.6251** | **$6.87 \times 10^{-14}$** | **458.0** | **$4.17 \times 10^{-14}$** | **0.8338** | **93 / 14** |
| Input Sensitivity vs Sobel | +0.0748 | 0.1043 | [+0.0548, +0.0948] | 7.4155 | $3.13 \times 10^{-11}$ | 625.0 | $1.97 \times 10^{-12}$ | 0.7169 | 91 / 16 |
| $D_4$ TTA vs Sobel | -0.0307 | 0.1281 | [-0.0552, -0.0061] | -2.4775 | $0.0148$ | 1750.0 | $0.00040$ | -0.2395 | 34 / 73 |
| **Hybrid vs Input Sensitivity** | **+0.0045** | 0.0344 | [-0.0020, +0.0111] | 1.3664 | $0.1747$ | **1783.0** | **$0.00059$** | 0.1321 | **71 / 36** |
| Hybrid vs $D_4$ TTA | +0.1100 | 0.1295 | [+0.0852, +0.1348] | 8.7840 | $3.03 \times 10^{-14}$ | 531.0 | $2.32 \times 10^{-13}$ | 0.8492 | 92 / 15 |

---

## 4. Secondary Endpoints: Sparsification and Error Retention

### 4.1 Area Under the Random Error Retention Gain Curve (AURG)

AURG is formally defined as the area between the random sparsification curve and the model sparsification curve:
$$\text{AURG} = \int_{0}^{0.95} \left( \text{MAE}_{\text{random}}(p) - \text{MAE}_{\text{model}}(p) \right) dp$$
integrated numerically using the trapezoidal rule across 20 uniform retention intervals $p \in [0.0, 0.95]$. Higher values indicate superior ability to isolate and remove high-error pixels early.

| Estimator | Mean AURG | SD | Median AURG | 95% CI | Paired Gain vs Sobel | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Hybrid** | **0.01593** | 0.00998 | **0.01535** | **[0.01401, 0.01784]** | **+0.00469** | **$2.96 \times 10^{-14}$** |
| Input Sensitivity | 0.01562 | 0.00992 | 0.01475 | [0.01372, 0.01752] | +0.00439 | $1.21 \times 10^{-11}$ |
| Sobel Gradient | 0.01124 | 0.00893 | 0.01110 | [0.00952, 0.01295] | Reference | — |
| $D_4$ TTA | 0.01082 | 0.00975 | 0.01054 | [0.00895, 0.01269] | -0.00041 | 0.6012 |

### 4.2 Reconstruction MAE Reduction Across Sparsification Levels

| Sparsification Level ($p$) | Hybrid MAE Reduction (%) | Sobel MAE Reduction (%) | Paired Gain ($\Delta$) | Paired $t$-stat ($p$-value) | Win Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10% Pixels Discarded** | **6.71%** $\pm$ 3.70% | 5.12% $\pm$ 3.20% | **+1.59%** | $8.07$ ($9.34 \times 10^{-13}$) | 84.11% (90/107) |
| **20% Pixels Discarded** | **11.27%** $\pm$ 6.43% | 8.43% $\pm$ 5.77% | **+2.84%** | $9.69$ ($2.75 \times 10^{-16}$) | **91.59% (98/107)** |
| **30% Pixels Discarded** | **15.08%** $\pm$ 8.95% | 11.16% $\pm$ 8.10% | **+3.92%** | $10.15$ ($1.42 \times 10^{-17}$) | 89.72% (96/107) |

---

## 5. Development vs Holdout Replication Analysis

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  DEVELOPMENT VS HOLDOUT REPLICATION                        │
├───────────────────────────────────┬───────────────────┬────────────────────┤
│ Metric                            │ Development (CV)  │ Holdout (Frozen)   │
├───────────────────────────────────┼───────────────────┼────────────────────┤
│ Number of Evaluated Tiles         │ 107               │ 107                │
│ Hybrid Mean Spearman $\rho$       │ 0.3048            │ 0.2800             │
│ Sobel Mean Spearman $\rho$        │ 0.2224            │ 0.2006             │
│ Paired Mean Gain ($\bar{\Delta}$) │ +0.0824           │ +0.0793            │
│ Relative Gain over Sobel          │ +37.03%           │ +39.53%            │
│ Paired $t$-test $p$-value         │ 3.20 × 10⁻¹¹      │ 6.87 × 10⁻¹⁴       │
│ Win Rate vs Sobel                 │ 77.57% (83/107)   │ 86.92% (93/107)    │
│ Cohen's $d$ Effect Size           │ 0.7428            │ 0.8338             │
└───────────────────────────────────┴───────────────────┴────────────────────┘
```

### Analysis of Generalization
1. **Absolute Correlation Shift:** Both Hybrid ($\rho: 0.3048 \to 0.2800$) and Sobel ($\rho: 0.2224 \to 0.2006$) exhibited a minor baseline decrease of $\approx 0.023$ correlation units on the holdout partition. This reflects natural geographic scene variability across independent partitions.
2. **Stability of the Treatment Effect:** The paired advantage ($\bar{\Delta}$) remained exceptionally stable: $+0.0824$ on development vs $+0.0793$ on holdout (a difference of only $0.0031$).
3. **Win Rate Increase:** The proportion of tiles where Hybrid outperformed Sobel increased from $77.57\%$ to $86.92\%$, while Cohen's $d$ increased from $0.74$ to $0.83$.

---

## 6. Runtime, Computational Cost, and Hardware Efficiency

- **Hardware Used:** Single workstation GPU (NVIDIA RTX 2000 Ada Generation, 16GB VRAM, CUDA).
- **Batching & Precision:** Single-tile processing, FP32 inference (`torch.inference_mode`).
- **Total Holdout Runtime:** **42.54 seconds** across all 107 tiles.
- **Latency Per Tile:** **0.398 seconds / tile** (includes 8 $D_4$ TTA forward passes, 6-strategy variance computation, integer registration search over $[-8, 8]^2$, metric computation, and compressed `.npz` storage).

---

## 7. Scientific Interpretation and Strict Boundaries

### Distinguishing Statistical Evidence from Practical Effect Size
- **Statistical Evidence:** The statistical tests demonstrate conclusively ($p < 10^{-13}$, Cohen's $d = 0.83$) that Hybrid Uncertainty ranks super-resolution reconstruction errors substantially better than a standard high-frequency spatial edge baseline (Sobel gradient).
- **Practical Effect Size:**
  - An absolute correlation of $\rho \approx 0.28$ indicates a **moderate error-ranking signal**, not deterministic error detection.
  - In sparsification workflows, discarding the 20% most uncertain pixels reduces reconstruction error by $\approx 11.3\%$. While superior to Sobel ($8.4\%$), significant residual error remains in the retained pixels.
  - Hybrid uncertainty functions effectively as an empirical risk screen, but cannot guarantee error-free predictions.

### Mandatory Non-Claims
1. **Not Calibrated Probabilities:** The outputs of $U_{\text{hybrid}}$ are standardized relative rank scores ($Z$-score combinations), not Bayesian posterior variances or calibrated probabilistic prediction intervals.
2. **Not Building-Level Accuracy:** This evaluation did not assess building footprint segmentation masks or vector accuracy. Claims regarding building detection or polygon precision are not supported by this experiment.
3. **Not Geodetic Alignment:** The registration policy utilizes an image-space heuristic ($\pm 8$ pixel integer shift minimizing MSE). This accounts for local scene translation but does not provide geodetic orthorectification or terrain displacement correction.
4. **No Claim of Broad Generalization:** Findings are strictly validated on the evaluated 107 holdout tiles (rural/suburban/urban scenes in the existing dataset). Generalization to tropical regions, arid deserts, steep topography, or different sensor constellations cannot be inferred without dedicated testing.

---

## 8. Artifact Deliverables Inventory

All outputs from this experiment have been generated and preserved under:
`results/satlas_sr_experiment_017_locked_holdout/`

1. `preflight_protocol.json`: Frozen preflight parameters, split verification, and normalization statistics.
2. `run_holdout_evaluation.py`: Exact Python code executing the frozen evaluation.
3. `per_tile_holdout_metrics.csv`: Complete table of 107 holdout tiles with all per-tile metrics.
4. `summary.json`: Complete statistical breakdown, secondary metrics, and replication comparisons.
5. `maps/`: 107 compressed `.npz` files containing full arrays (`u_hybrid`, `u_tta`, `u_input`, `u_grad`, `err_comp`, `err_uncomp`, `shift_dx`, `shift_dy`).
6. `run_command_record.txt`: Execution log with timestamps, commit hash, hardware specs, and command records.
7. `EXP017_LOCKED_HOLDOUT_REPORT.md`: This comprehensive report.

**Protocol Status:** The holdout evaluation is concluded and frozen. No further tuning or rerunning on the holdout partition will be performed.
