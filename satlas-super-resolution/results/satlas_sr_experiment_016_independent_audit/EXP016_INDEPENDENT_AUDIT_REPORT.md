# EXP-016: Independent Statistical and Leakage Audit of EXP-015

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Audited Artifacts**: `results/satlas_sr_experiment_015_hybrid_uncertainty/`  
**Audit Scope**: Strict Read-Only Independent Verification of 107 Development Tiles (Zero Holdout Access)  
**Audit Status**: COMPLETED (10/10 Verification Checks Passed)  
**Audit Verdict**: **READY FOR LOCKED HOLDOUT**  
**Date**: 2026-10-23  

---

## 1. Executive Summary & Audit Verdict

This audit performs an independent numerical verification and methodology audit of **EXP-015 (Hybrid Uncertainty Feasibility and Development-Only Comparison)** before opening the locked holdout partition.

### Audit Verdict: **READY FOR LOCKED HOLDOUT**
- **Zero Data Leakage Verified**: The 107 development tiles were confirmed to have zero overlap with the 107 locked holdout tiles or the 42 quarantined tiles.
- **Cross-Validation Integrity Verified**: Standardization parameters ($\mu, \sigma$) and hybrid blending weights ($\alpha^*$) were strictly computed using training folds only. Validation folds remained completely out-of-fold.
- **Spatial Buffer Enforced**: The minimum Chebyshev distance between any two tiles in *different* cross-validation folds is **12 tiles ($> 3.6\text{ km}$)**. Zero spatial clusters were split across folds.
- **Exact Numerical Match**: All reported means, standard deviations, confidence intervals, medians, paired differences, $t$-tests, and Wilcoxon signed-rank tests match the raw per-tile data to full floating-point precision ($|\Delta| < 10^{-6}$).
- **Key Scientific Nuance**: The hybrid estimator's statistically significant superiority over the Sobel gradient baseline ($\bar{\Delta} = +0.0822$, $p = 3.20 \times 10^{-11}$, Cohen's $d = 0.716$) is **overwhelmingly driven by Sentinel-2 input-selection sensitivity** ($\bar{\Delta}_{\text{input vs Sobel}} = +0.0756$, accounting for $\sim 92\%$ of the gain). Fusing $D_4$ TTA adds a modest but statistically significant incremental boost of **$+0.00665$** ($+2.24\%$, $p = 0.0001$), winning on **$72.0\%$ of tiles**.

---

## 2. Task 1: Independent Numerical Recomputation

Data was independently derived from [`results/satlas_sr_experiment_015_hybrid_uncertainty/per_tile_oof_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/per_tile_oof_metrics.csv) without relying on summary values:
- Total rows verified: Exactly **107**.
- Unique tile IDs: Exactly **107** (0 duplicates, 0 missing).
- Non-finite values (`NaN`, `Inf`): Exactly **0**.
- Holdout / Quarantine contamination: Exactly **0**.

### 2.1. Recomputed Distribution Statistics ($N=107$ Tiles)

| Metric Family | Method / Estimator | Mean ($\bar{x}$) | Sample SD ($s$) | SEM | 95% Confidence Interval | Median | IQR | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spearman $\rho_{\text{comp}}$** | Constant Baseline | $0.0000$ | $0.0000$ | $0.0000$ | $[0.0000, 0.0000]$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |
| Spearman $\rho_{\text{comp}}$ | $D_4$ TTA Uncertainty | $0.1910$ | $0.1651$ | $0.0160$ | $[0.1594, 0.2226]$ | $0.1896$ | $0.1782$ | $-0.4421$ | $+0.5889$ |
| Spearman $\rho_{\text{comp}}$ | Sobel Gradient Baseline | $0.2220$ | $0.1638$ | $0.0158$ | $[0.1906, 0.2534]$ | $0.2324$ | $0.1763$ | $-0.4125$ | $+0.5797$ |
| Spearman $\rho_{\text{comp}}$ | Input Sensitivity ($U_{\text{input}}$) | $0.2976$ | $0.1517$ | $0.0147$ | $[0.2685, 0.3266]$ | $0.3095$ | $0.1645$ | $-0.2261$ | $+0.7044$ |
| **Spearman $\rho_{\text{comp}}$** | **Hybrid Estimator ($U_{\text{hybrid}}$ OOF)** | **$0.3042$** | $0.1538$ | $0.0149$ | **$[0.2747, 0.3337]$** | **$0.3081$** | $0.1637$ | $-0.2707$ | $+0.6536$ |
| **AURG** | Constant Baseline | $0.0000$ | $0.0000$ | $0.0000$ | $[0.0000, 0.0000]$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |
| AURG | $D_4$ TTA Uncertainty | $0.0125$ | $0.0107$ | $0.0010$ | $[0.0104, 0.0145]$ | $0.0115$ | $0.0102$ | $-0.0164$ | $+0.0764$ |
| AURG | Sobel Gradient Baseline | $0.0122$ | $0.0094$ | $0.0009$ | $[0.0104, 0.0140]$ | $0.0121$ | $0.0119$ | $-0.0178$ | $+0.0400$ |
| AURG | Input Sensitivity ($U_{\text{input}}$) | $0.0161$ | $0.0089$ | $0.0009$ | $[0.0144, 0.0178]$ | $0.0159$ | $0.0118$ | $-0.0098$ | $+0.0391$ |
| **AURG** | **Hybrid Estimator ($U_{\text{hybrid}}$ OOF)** | **$0.0170$** | $0.0093$ | $0.0009$ | **$[0.0152, 0.0188]$** | **$0.0170$** | $0.0122$ | $-0.0121$ | $+0.0393$ |

*Archived in [`recomputed_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_metrics.csv).*

---

### 2.2. Recomputed Paired Hypothesis Tests & Effect Sizes

To eliminate ambiguity, relative gains were computed with explicit denominators:
$$\text{Relative Gain (\%)} = \frac{\bar{x}_{\text{method1}} - \bar{x}_{\text{method2}}}{\bar{x}_{\text{method2}}} \times 100$$

| Paired Comparison | Mean Diff ($\bar{\Delta}$) | Sample $s_{\Delta}$ | 95% CI of $\bar{\Delta}$ | Cohen's $d$ | Explicit Denominator | Rel. Gain | Win / Loss | Paired $t$ | Paired $t$ $p$ | Wilcoxon $W$ | Wilcoxon $p$ | Bonferroni $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid vs Sobel** | **$+0.0822$** | $0.1147$ | $[+0.0602, +0.1042]$ | **$0.7164$** | $\bar{\rho}_{\text{Sobel}} = 0.222005$ | **$+37.03\%$** | **$96$ / $11$** | $7.4107$ | $3.20 \times 10^{-11}$ | $336.0$ | $2.11 \times 10^{-15}$ | **$8.44 \times 10^{-15}$** |
| **Hybrid vs TTA** | **$+0.1132$** | $0.1343$ | $[+0.0875, +0.1390]$ | **$0.8429$** | $\bar{\rho}_{\text{TTA}} = 0.190989$ | **$+59.28\%$** | **$89$ / $18$** | $8.7191$ | $4.24 \times 10^{-14}$ | $459.0$ | $4.27 \times 10^{-14}$ | **$1.71 \times 10^{-13}$** |
| **Hybrid vs Input** | **$+0.0067$** | $0.0275$ | $[+0.0014, +0.0119]$ | **$0.2423$** | $\bar{\rho}_{\text{Input}} = 0.297557$ | **$+2.24\%$** | **$77$ / $30$** | $2.5064$ | $0.0137$ | $1610.0$ | $0.0001$ | **$0.0003$** |
| **Input vs Sobel** | **$+0.0756$** | $0.1231$ | $[+0.0520, +0.0991]$ | **$0.6137$** | $\bar{\rho}_{\text{Sobel}} = 0.222005$ | **$+34.03\%$** | **$92$ / $15$** | $6.3481$ | $5.52 \times 10^{-9}$ | $507.0$ | $1.33 \times 10^{-13}$ | **$5.31 \times 10^{-13}$** |

*Archived in [`recomputed_paired_tests.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv).*

---

## 3. Task 2: Audit of Spatial Cross-Validation & Spatial Dependence

### 3.1. Inter-Fold Buffer Verification
- **Audit Question**: Was the claimed Chebyshev distance buffer ($\le 10$ tiles) strictly enforced between folds?
- **Verification Evidence**:
  - Distance matrix computation between all pairs of tiles across different folds demonstrated that the **minimum Chebyshev distance between any two tiles in different folds is 12 tiles ($> 3.6\text{ km}$)**.
  - The closest pair across folds is `38661_49188` (Fold 2) and `38673_49176` (Fold 4) with distance $\Delta x = 12, \Delta y = 12$.
  - Exactly **0 pairs** within Chebyshev distance $\le 10$ tiles were split across folds.
  - All 4 multi-tile clusters in the development set (`26847_48219`/`26847_48228`, `35744_54906`/`35754_54911`, `35797_54930`/`35806_54920`, `37523_49502`/`37524_49502`) are **100% contained within individual folds**.

### 3.2. Statistical Unit of Analysis & Degrees of Freedom
- **Unit of Analysis**: The inferential tests operate strictly on the **per-tile aggregate metric** ($N=107$). Individual pixels ($14,400$ per tile) are never treated as independent samples.
- **Spatial Independence**: Across the 107 tiles, there are **103 independent geographic spatial clusters** distributed nationwide. Even when adjusting degrees of freedom conservatively from $df=106$ to $df=102$, the paired $t$-statistic ($t = 7.41$) remains significant at $p = 4.1 \times 10^{-11}$.

---

## 4. Task 3: Data Leakage and Preprocessing Isolation Audit

Direct code inspection of [`run_exp015_cv.py:L140-L170`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/run_exp015_cv.py) confirmed strict isolation:
1. **Standardization Parameters ($\mu, \sigma$)**:
   - `train_tta_vals` and `train_input_vals` are concatenated strictly from `train_tiles`.
   - Validation-fold tiles are completely excluded from computing $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
2. **Weight Optimization ($\alpha$)**:
   - Grid search across $\alpha \in [0.0, 1.0]$ evaluated Spearman correlation strictly across `train_tiles`.
   - The selected $\alpha^*$ was frozen and applied to the validation fold.
3. **No Reference Error Leakage**:
   - Component map standardization does not utilize reference images or error maps; only the final evaluation compares against NAIP residuals.
4. **Registration Policy Consistency**:
   - All methods (Constant, Sobel, TTA, Input, Hybrid) were evaluated on the identical $120 \times 120$ interior subwindow under the identical integer shift $[\Delta x^*, \Delta y^*]$.

---

## 5. Task 4: Uncertainty Maps & Component Contribution Audit

### 5.1. What Drives the Hybrid's Improvement?
The audit decomposes the improvement over the Sobel gradient baseline into two components:
1. **Input Sensitivity Contribution**:
   - $\bar{\Delta}_{\text{input vs Sobel}} = +0.075552$ ($p = 1.33 \times 10^{-13}$).
   - **Accounts for $91.9\%$ of the hybrid's total gain over Sobel**.
2. **TTA Orientation Sensitivity Contribution**:
   - $\bar{\Delta}_{\text{hybrid vs input}} = +0.006653$ ($p = 0.00010$, Bonferroni $p = 0.00030$).
   - **Accounts for $8.1\%$ of the hybrid's total gain over Sobel**.
   - Wins on **$77$ out of $107$ tiles ($72.0\%$)**, with 0 ties and 30 losses.
   - Cohen's $d = 0.242$ (small but statistically robust effect size).

**Takeaway**: Multi-temporal observation sensitivity ($U_{\text{input}}$) is the primary driver of error-ranking capability in multi-frame satellite super-resolution. TTA provides a secondary, fine-grained structural regularization ($\alpha^* \approx 0.19$) that reliably refines edge boundaries.

---

## 6. Task 5: Reassessment of Statistical Claims & Effect Sizes

1. **"Statistically Significant Out-Of-Fold Advantage"**:
   - **SUPPORTED**: The hybrid estimator demonstrates a large effect size over Sobel (Cohen's $d = 0.716$, paired $t = 7.41$, $p = 3.20 \times 10^{-11}$). Win rate is **$89.7\%$ (96/107 tiles)**.
2. **Multiple Comparison Corrections**:
   - Correcting across all 4 pairwise comparisons via Bonferroni adjustment preserves significance ($p_{\text{Bonf}} < 10^{-12}$ for Hybrid vs Sobel; $p_{\text{Bonf}} = 0.00030$ for Hybrid vs Input).
3. **Practical Meaningfulness**:
   - In sparsification analysis, masking the top 20% most uncertain pixels reduces MAE by **$12.98\%$** for the hybrid (compared to $9.95\%$ for Sobel and $8.48\%$ for TTA).
   - This represents a meaningful, monotonic error reduction that outperforms heuristic spatial edge detectors.

---

## 7. Deliverables & Audit Manifest

All artifacts have been created under [`results/satlas_sr_experiment_016_independent_audit/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/):
1. [`EXP016_INDEPENDENT_AUDIT_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/EXP016_INDEPENDENT_AUDIT_REPORT.md): This report.
2. [`recomputed_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_metrics.csv): Complete independent derivation of distribution statistics for all 5 methods.
3. [`recomputed_paired_tests.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv): Complete independent paired comparisons, effect sizes, win/loss tallies, and hypothesis tests.
4. [`audit_checks.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/audit_checks.json): 10/10 passing verification checks.
5. [`run_command_record.txt`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/run_command_record.txt): Transparent execution trace.

---

## 8. Final Recommendation

### **READY FOR LOCKED HOLDOUT**

The EXP-015 methodology, data leakage protections, spatial cross-validation buffers, and numerical claims have been thoroughly audited and independently reproduced with zero defects. The hybrid uncertainty estimator ($U_{\text{hybrid}}$ with frozen parameters $\alpha^* = 0.19$, $\mu_{\text{tta}} = 0.0831$, $\sigma_{\text{tta}} = 0.0514$, $\mu_{\text{input}} = 0.0573$, $\sigma_{\text{input}} = 0.0412$) is fully specified and ready for frozen evaluation on the **locked holdout partition ($N=107$ tiles)**.

Per protocol, **the holdout partition remains locked and untouched**. Would you like to approve executing the final frozen evaluation on the locked holdout?
