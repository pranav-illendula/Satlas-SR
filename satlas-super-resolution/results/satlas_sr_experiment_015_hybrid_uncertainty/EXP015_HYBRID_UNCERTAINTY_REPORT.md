# EXP-015: Hybrid Uncertainty Feasibility and Development-Only Comparison Report

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe` (PyTorch 2.1.0+cu121, NVIDIA RTX 2000 Ada Generation)  
**Evaluation Scope**: 107-Tile Development Partition Only (5-Fold Spatially Grouped Out-Of-Fold CV)  
**Holdout Status**: 107-Tile Holdout & 42-Tile Quarantine Strictly Locked and Untouched  
**Date**: 2026-10-23  

---

## 1. Executive Summary

EXP-015 investigated whether combining **$D_4$ Test-Time Augmentation (TTA) uncertainty** with **Sentinel-2 input-selection sensitivity** provides a statistically significant, reproducible advantage over the simple Sobel image-gradient baseline, which had matched or slightly outperformed TTA alone in EXP-014.

### Key Empirical Findings:
1. **Technical Feasibility Confirmed**:
   - Audit of EXP-011 predictions revealed complete, uncorrupted $128 \times 128 \times 3$ predictions saved across all 7 strategies (S0–S6) on disk for all 107 development tiles.
   - Per-pixel input-selection sensitivity ($U_{\text{input}}$) was constructed from 6 distinct temporal observation draws (S0, S1, S3, S4, S5, S6) without broadcast approximations.
2. **Input Sensitivity Alone Outperforms Sobel Gradient**:
   - Sentinel-2 input-selection sensitivity achieves a mean Spearman rank correlation of **$\bar{\rho} = 0.2976 \pm 0.1517$**, significantly outperforming the Sobel gradient baseline (**$\bar{\rho} = 0.2220 \pm 0.1638$**, paired difference $\bar{\Delta} = +0.0756$, paired $t = 6.35$, $p = 5.52 \times 10^{-9}$, Wilcoxon $p = 1.33 \times 10^{-13}$).
3. **Hybrid Estimator Out-Of-Fold Superiority**:
   - Evaluated under a strict **5-fold spatially grouped Out-Of-Fold (OOF) cross-validation** design (zero data leakage), the hybrid estimator $U_{\text{hybrid}} = \alpha^* Z(U_{\text{tta}}) + (1 - \alpha^*) Z(U_{\text{input}})$ selected optimal $\alpha^* \in [0.15, 0.20]$ (mean $0.19$) across all training folds.
   - Out-of-fold performance achieved:
     * **Spearman Rank Correlation**: **$\bar{\rho} = 0.3042 \pm 0.1538$** (95% CI $[0.2747, 0.3337]$).
     * **Area Under Random Gain (AURG)**: **$0.0170 \pm 0.0093$** (vs. Sobel $0.0122$, vs. TTA $0.0125$).
     * **MAE Reduction at 20% Sparsification**: **$12.98 \pm 6.32\%$** (vs. Sobel $9.95\%$, vs. TTA $8.48\%$).
   - In paired statistical testing against the Sobel gradient baseline, the hybrid demonstrates a **statistically significant advantage**:
     * Paired difference: $\bar{\Delta} = \mathbf{+0.0822}$ ($+37.0\%$ relative improvement).
     * Paired Student's $t = 7.41$, **$p = 3.20 \times 10^{-11}$**.
     * Paired Wilcoxon signed-rank **$p = 2.11 \times 10^{-15}$**.
4. **Primary Diagnostic Insight**:
   - $U_{\text{tta}}$ and $U_{\text{input}}$ exhibit low mutual rank correlation ($\rho \approx 0.19 - 0.35$), confirming that they capture complementary error mechanisms. TTA captures geometric/rotational ambiguity along sharp edges, while input sensitivity captures temporal/atmospheric observation ambiguity across satellite revisits.

---

## 2. Technical Feasibility Audit

| Modality / Quantity | Representation on Disk | Source Artifact | Physical Meaning & Units |
| :--- | :--- | :--- | :--- |
| **$U_{\text{tta}}$ (TTA Sensitivity)** | Per-pixel float32 map ($128 \times 128$) | EXP-014 `maps/*.npz` | Sample standard deviation across 8 inverse $D_4$ transforms ($[0, 1]$ reflectance). |
| **$U_{\text{input}}$ (Input Sensitivity)** | 6 saved full predictions ($128 \times 128 \times 3$) | EXP-011 `predictions/{S0..S6}/` | Sample standard deviation across 6 distinct frame draws ($[0, 1]$ reflectance). |
| **$U_{\text{grad}}$ (Sobel Gradient)** | Per-pixel float32 map ($120 \times 120$) | Computed on S0 luminance | 2D spatial gradient magnitude (pixel intensity difference / unit distance). |
| **$E_{\text{comp}}$ (Reconstruction Error)** | Per-pixel float32 map ($120 \times 120$) | EXP-014 `maps/*.npz` | Absolute difference against NAIP under optimal integer translation $[\Delta x^*, \Delta y^*]$. |

All predictions share identical dimensions ($128 \times 128 \times 3$), coordinate orientations, and $120 \times 120$ interior evaluation subwindows (excluding 4-pixel borders).

---

## 3. Mathematical Definition of the Hybrid Estimator

To combine $U_{\text{tta}}$ and $U_{\text{input}}$ without scale mismatch, fold-level standardization was applied:
$$Z(U) = \frac{U - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ were computed strictly across all training-fold pixels. The hybrid score is:
$$U_{\text{hybrid}}(\alpha) = \alpha \cdot Z(U_{\text{tta}}) + (1 - \alpha) \cdot Z(U_{\text{input}})$$
The optimal blending weight $\alpha^*$ was selected via grid search on training folds ($\alpha \in [0, 1]$, step $0.05$) to maximize training Spearman correlation.

---

## 4. Spatially Grouped 5-Fold Cross-Validation Design

To prevent overfitting or tuning leakage on the 107 development tiles:
1. Spatial clustering with Chebyshev distance $\le 10$ tiles ($\sim 3.3\text{ km}$ buffer) grouped nearby scenes into 103 independent clusters.
2. Clusters were partitioned into 5 balanced folds ($[21, 21, 21, 22, 22]$ tiles).
3. Zero spatial leakage exists across folds (clusters never split).
4. For each fold $k \in \{0, \dots, 4\}$:
   - Normalization statistics and $\alpha^*$ were selected on the remaining 4 folds (~85–86 tiles).
   - Evaluation was performed strictly on the held-out fold $k$ (~21–22 tiles).
   - Out-of-fold metrics were aggregated across all 107 tiles.

---

## 5. Statistical Results & Hypothesis Tests ($N=107$ Tiles)

### 5.1. Performance Summary Table

| Estimator / Baseline | Spearman $\rho_{\text{comp}}$ (Mean ± SD) | 95% CI of Mean $\rho$ | AURG (Mean ± SD) | MAE Red. @ 10% | MAE Red. @ 20% | MAE Red. @ 30% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Baseline** | $0.0000 \pm 0.0000$ | $[0.000, 0.000]$ | $0.0000 \pm 0.0000$ | $0.00\%$ | $0.00\%$ | $0.00\%$ |
| **$D_4$ TTA Uncertainty** | $0.1910 \pm 0.1651$ | $[0.1594, 0.2226]$ | $0.0125 \pm 0.0107$ | $4.49\%$ | $8.48\%$ | $11.75\%$ |
| **Sobel Gradient Baseline** | $0.2220 \pm 0.1638$ | $[0.1906, 0.2534]$ | $0.0122 \pm 0.0094$ | $5.39\%$ | $9.95\%$ | $13.62\%$ |
| **Input Sensitivity ($U_{\text{input}}$)** | $0.2976 \pm 0.1517$ | $[0.2685, 0.3266]$ | $0.0161 \pm 0.0089$ | $6.92\%$ | $12.56\%$ | $17.15\%$ |
| **Hybrid Estimator ($U_{\text{hybrid}}$ OOF)** | **$0.3042 \pm 0.1538$** | **$[0.2747, 0.3337]$** | **$0.0170 \pm 0.0093$** | **$7.18\%$** | **$12.98\%$** | **$17.65\%$** |

### 5.2. Paired Hypothesis Tests vs. Sobel Gradient Baseline

- **Primary Metric (Spearman Rank Correlation)**:
  - Mean paired difference $\bar{\Delta} = \bar{\rho}_{\text{hybrid}} - \bar{\rho}_{\text{grad}} = \mathbf{+0.0822}$ ($+37.0\%$).
  - Paired Student's $t = 7.411$, **$p = 3.20 \times 10^{-11}$**.
  - Paired Wilcoxon signed-rank $W = 336.0$, **$p = 2.11 \times 10^{-15}$**.
- **Secondary Metric (Area Under Random Gain — AURG)**:
  - Mean paired difference: $+0.00481$ ($+39.5\%$ gain over Sobel).
  - Wilcoxon signed-rank $p = 3.31 \times 10^{-15}$.
- **Secondary Metric (20% Sparsification MAE Reduction)**:
  - Mean improvement: $+3.03\%$ absolute reduction ($12.98\%$ vs $9.95\%$).

---

## 6. Metric Implementation Unit Tests Summary

Synthetic verification was executed in [`metric_unit_tests.txt`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/metric_unit_tests.txt):
1. **Case 1 (Perfect Ranking $U = E$)**: $\rho = 1.000000$, $\text{AUSE} = 0.000000$, $\text{AURG} = \text{AUSE}_{\text{random}}$ (PASS).
2. **Case 2 (Reverse Ranking $U = -E$)**: $\rho = -1.000000$, $\text{AURG} < 0$, negative MAE reduction (PASS).
3. **Case 3 (Constant Uncertainty $U = c$)**: $\rho = 0.000000$ (tie handling), $\text{AUSE} = \text{AUSE}_{\text{random}}$, $\text{AURG} = 0.000000$ (PASS).
4. **Case 4 (Random Uncertainty $U \sim \mathcal{U}(0, 1)$)**: $|\rho| < 0.02$, $|\text{AURG}| < 0.001$ (PASS).
5. **Case 5 (Edge-Correlated, Error-Independent)**: $\rho = 0.001064$, $\text{AURG} \approx 0$ (PASS).

---

## 7. Strict Scientific Non-Claims

1. **Not Calibrated Probabilistic Uncertainty**: The hybrid score is an uncalibrated error-ranking metric. It does not provide conformal prediction bounds or valid posterior coverage intervals.
2. **Zero Building-Level Accuracy Claims**: Authoritative building annotations remain unavailable locally. High uncertainty indicates model/temporal disagreement; it does not verify building omissions or structural fidelity.
3. **Generalization Scope**: These results apply strictly to the evaluated development sample under 2019 NAIP aerial references.

---

## 8. Conclusion & Follow-Up Recommendation

Unlike $D_4$ TTA alone (which failed to outperform image gradient in EXP-014), the **Hybrid Estimator demonstrated a statistically significant, consistent advantage over the Sobel gradient baseline across all 5 spatial cross-validation folds ($p < 10^{-10}$)**.

### Recommendation Regarding Holdout Evaluation:
The hybrid estimator is now scientifically justified for evaluation on the locked holdout partition ($N=107$ tiles). However, per protocol, **the holdout remains strictly locked**. Approval from the user is required before initiating holdout evaluation.
