# EXP-014: D₄ Test-Time Augmentation Uncertainty Prototype and Development-Set Evaluation Report

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe` (PyTorch 2.1.0+cu121, NVIDIA RTX 2000 Ada Generation)  
**Experiment Mode**: Development Split Evaluation Only (Locked Holdout Untouched)  
**Checkpoint Used**: `experiments/esrgan_8S2.pth` (`params_ema`, SHA256 `6e0767cb...`)  
**Status**: COMPLETED  
**Date**: 2026-10-23  

---

## 1. Executive Summary

This experiment implements and evaluates a **$D_4$ Test-Time Augmentation (TTA)** uncertainty estimation prototype for the frozen Satlas-SR ESRGAN baseline, following the staged protocol designed in EXP-013. The investigation was conducted strictly on the **development partition of 107 tiles**, keeping the 107-tile holdout partition and 42 quarantined tiles completely untouched.

### Core Empirical Findings:
1. **Mathematical Soundness of $D_4$ Transforms Verified**:
   - The 8 transformations of the discrete dihedral group $D_4$ (4 orthogonal rotations $\times$ 2 reflections) were implemented using pure tensor transposition and flips, requiring **zero continuous spatial interpolation**.
   - Synthetic round-trip unit tests achieved **exact zero difference** ($\max |\Delta| = 0.000000$) across all 8 symmetries.
   - Pilot tests on development tiles verified exact correspondence with the frozen baseline disk prediction ($\max |\Delta| = 0$).
2. **Statistically Significant Superiority Over Random Chance**:
   - On the 107 development tiles, the Spearman rank correlation between $D_4$ TTA uncertainty and registration-compensated absolute reconstruction error is **$\rho = 0.1910 \pm 0.1651$** ($95\%\text{ CI } [0.1594, 0.2226]$), compared to **$\rho = -0.0008 \pm 0.0087$** for uniform random uncertainty.
   - The Area Under Random Gain (AURG) is **$0.0125 \pm 0.0107$**, with a paired Wilcoxon signed-rank test against zero yielding **$p = 9.36 \times 10^{-17}$**.
   - Progressively sparsifying (masking) the top 20% most uncertain pixels reduces the residual Mean Absolute Error (MAE) by **$8.48 \pm 6.63\%$** (median $7.58\%$, max $35.6\%$).
3. **The Heuristic Baseline Challenge (TTA vs. Image Gradient)**:
   - Benchmarking against a simple 2D Sobel gradient magnitude baseline ($U_{\text{gradient}}$) reveals that local image gradient correlates slightly *better* with reconstruction error than the deep TTA ensemble ($\bar{\rho}_{\text{grad}} = 0.2220 \pm 0.1640$ vs. $\bar{\rho}_{\text{TTA}} = 0.1910 \pm 0.1651$, paired $t$-test $p = 0.0157$, Wilcoxon $p = 0.0263$).
   - Both methods achieve comparable sparsification gain ($\text{AURG}_{\text{TTA}} = 0.01247$ vs. $\text{AURG}_{\text{grad}} = 0.01219$).
   - **Key Scientific Insight**: Much of what $D_4$ TTA uncertainty captures is high-frequency edge complexity. Because super-resolution errors naturally concentrate along sharp high-contrast edges due to sensor point spread functions and sub-pixel shifts, simple texture complexity acts as an equally potent error predictor without requiring 8 forward passes.
4. **Registration Compensation Impact**:
   - Mean residual integer shift on the development tiles was $1.35 \pm 1.20\text{ pixels}$ (median $1.0\text{ px}$).
   - Uncompensated Spearman correlation was slightly higher ($\rho = 0.2009$) than compensated correlation ($\rho = 0.1910$) because uncompensated registration jitter artificially inflates errors along high-contrast boundaries, where TTA variance is also elevated.

---

## 2. Experimental Setup, Partitioning, and Leakage Prevention

### 2.1. Spatial Clustering and Partition Integrity
To eliminate risk of spatial data leakage across neighboring geographic scenes:
1. Spatial coordinates $(X, Y)$ of the 214 reliable tiles from EXP-013 (`results/satlas_sr_spatial_audit/registration_validation.csv`) were mapped onto the EPSG:3857 Slippy Map grid ($z=17$).
2. Across all 214 reliable tiles, exactly **one pair of directly adjacent tiles** exists (`37523_49502` and `37524_49502`).
3. Spatial graph components were constructed with a **Chebyshev distance threshold of 10 tiles** ($\sim 3.3\text{ km}$ buffer). All clustered tiles within this buffer were constrained to remain in the same partition.
4. The resulting 206 clusters were partitioned deterministically into:
   - **Development Split**: Exactly **107 tiles** (used for EXP-014 evaluation).
   - **Locked Holdout Split**: Exactly **107 tiles** (locked, untouched).
   - **Quarantined Set**: Exactly **42 tiles** (excluded due to low correlation or $>15\text{ px}$ shift).
5. **Spatial Exclusion Bound**: The minimum Chebyshev distance between any development tile and any holdout tile is **11 tiles ($> 3.3\text{ km}$)**. Zero spatial leakage exists across the boundary.
6. Frozen partition archived in [`results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json).

---

## 3. Implementation and Verification of $D_4$ Group Transformations

### 3.1. Discrete Dihedral Symmetry Formulation
The dihedral group $D_4$ of the square contains 8 elements, parameterized by horizontal reflection $f \in \{0, 1\}$ and counter-clockwise rotation $k \in \{0, 1, 2, 3\}$:
$$\mathcal{T}_{D_4} = \{T_{(k, f)} \mid f \in \{0, 1\}, k \in \{0, 1, 2, 3\}\}$$

- **Forward Transformation**:
  $$\text{If } f = 1: X \leftarrow \text{flip}(X, \text{dim}=-1); \quad X \leftarrow \text{rot90}(X, k, \text{dims}=(-2, -1))$$
- **Exact Inverse Transformation**:
  $$\tilde{Y} \leftarrow \text{rot90}(Y, (4 - k) \pmod 4, \text{dims}=(-2, -1)); \quad \text{If } f = 1: \tilde{Y} \leftarrow \text{flip}(\tilde{Y}, \text{dim}=-1)$$
- **Zero Interpolation**: Implemented strictly via memory stride manipulation and index flipping (`torch.rot90`, `torch.flip`). No bilinear or bicubic resampling is performed ($T^{-1}(T(I)) \equiv I$).

### 3.2. Verification Milestones
1. **Synthetic Unit Test** ([`test_d4.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/scratch/test_d4.py)):
   - Evaluated on non-symmetric synthetic float32 arrays ($32 \times 32$).
   - Max difference: **$0.000000$** for all 8 transformations.
   - All 8 transformed arrays demonstrated unique byte representations.
2. **Development Pilot Test** ([`run_pilot_tta.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/scratch/run_pilot_tta.py)):
   - Executed on 3 development tiles (`20890_46801`, `21068_45692`, `22202_46519`).
   - Match with frozen disk prediction: $\max |\Delta| = 0$ (exact integer identity).
   - Match on $T_0$ identity: $\max |\Delta| = 0.000000$.
   - Deterministic repeatability: $\max |\Delta| = 0.000000$.
   - Validated inverse geometry across all 8 orientations.

---

## 4. Full Development Split Statistical Evaluation

The 107 development tiles were processed using [`run_dev_evaluation.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/run_dev_evaluation.py). Total execution time was **$60.06\text{ seconds}$** on an NVIDIA RTX 2000 Ada GPU ($\sim 0.56\text{ s/tile}$, 8 passes per tile).

### 4.1. Primary Evaluation Metrics Summary ($N=107$ Tiles)

| Metric | Candidate / Baseline Method | Mean ($\bar{x}$) | Sample SD ($s$) | SEM | 95% Confidence Interval | Median | IQR | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spearman Rank Correlation ($\rho_{\text{comp}}$)** | **$D_4$ TTA Uncertainty ($U_{\text{TTA}}$)** | **$0.1910$** | $0.1651$ | $0.0160$ | $[0.1594, 0.2226]$ | $0.1896$ | $0.1782$ | $-0.4421$ | $+0.5889$ |
| Spearman Rank Correlation ($\rho_{\text{uncomp}}$) | $D_4$ TTA (Uncompensated) | $0.2009$ | $0.1624$ | $0.0157$ | $[0.1697, 0.2320]$ | $0.2019$ | $0.1750$ | $-0.4417$ | $+0.5889$ |
| Spearman Rank Correlation ($\rho_{\text{comp}}$) | Sobel Image Gradient ($U_{\text{grad}}$) | **$0.2220$** | $0.1640$ | $0.0159$ | $[0.1906, 0.2534]$ | $0.2323$ | $0.1768$ | $-0.4156$ | $+0.5801$ |
| Spearman Rank Correlation ($\rho_{\text{comp}}$) | Random Baseline ($U_{\text{rand}}$) | $-0.0008$ | $0.0087$ | $0.0008$ | $[-0.0025, +0.0009]$ | $-0.0007$ | $0.0114$ | $-0.0242$ | $+0.0231$ |
| **Area Under Random Gain (AURG)** | **$D_4$ TTA Uncertainty ($U_{\text{TTA}}$)** | **$0.0125$** | $0.0107$ | $0.0010$ | $[0.0104, 0.0145]$ | $0.0115$ | $0.0102$ | $-0.0164$ | $+0.0764$ |
| Area Under Random Gain (AURG) | Sobel Image Gradient ($U_{\text{grad}}$) | $0.0122$ | $0.0094$ | $0.0009$ | $[0.0104, 0.0140]$ | $0.0122$ | $0.0120$ | $-0.0180$ | $+0.0401$ |
| **MAE Reduction at 20% Removal (%)** | **$D_4$ TTA Uncertainty ($U_{\text{TTA}}$)** | **$8.48\%$** | $6.63\%$ | $0.64\%$ | $[7.21\%, 9.75\%]$ | $7.58\%$ | $5.67\%$ | $-13.13\%$ | $+35.60\%$ |
| Registration Shift Magnitude (px) | Integer Search in $[-8, 8]$ | $1.35\text{ px}$ | $1.20\text{ px}$ | $0.12\text{ px}$ | $[1.12, 1.58]\text{ px}$ | $1.00\text{ px}$ | $0.41\text{ px}$ | $0.00\text{ px}$ | $5.66\text{ px}$ |

*All statistics compiled directly from [`per_tile_uncertainty_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/per_tile_uncertainty_metrics.csv) and archived in [`summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/summary.json).*

---

## 5. Statistical Hypothesis Tests & Paired Comparisons

### 5.1. Test 1: Is $D_4$ TTA Uncertainty Superior to Random Chance?
- **Null Hypothesis ($H_0$)**: $\text{AURG}_{\text{TTA}} \le 0$ (uncertainty provides no sparsification benefit over random removal).
- **Alternative Hypothesis ($H_1$)**: $\text{AURG}_{\text{TTA}} > 0$.
- **Statistical Test**: One-tailed Wilcoxon signed-rank test on 107 paired tiles.
- **Result**:
  - Test statistic $W = 5537.0$
  - **$p\text{-value} = 9.36 \times 10^{-17}$** ($p \ll 0.001$).
- **Verdict**: **REJECT $H_0$**. $D_4$ TTA uncertainty significantly outperforms uninformative random ranking at identifying reconstruction error.

### 5.2. Test 2: Does $D_4$ TTA Outperform Simple Image Gradient?
- **Null Hypothesis ($H_0$)**: $\mu_{\rho(\text{TTA})} = \mu_{\rho(\text{grad})}$ (no difference in rank correlation).
- **Alternative Hypothesis ($H_1$)**: $\mu_{\rho(\text{TTA})} \neq \mu_{\rho(\text{grad})}$.
- **Statistical Tests**:
  - Paired Student's $t$-test: $t = -2.455$, **$p = 0.0157$**.
  - Paired Wilcoxon signed-rank test: $W = 2174.0$, **$p = 0.0263$**.
  - Mean paired difference: $\bar{\Delta} = \bar{\rho}_{\text{TTA}} - \bar{\rho}_{\text{grad}} = \mathbf{-0.0310}$.
- **Verdict**: **$D_4$ TTA does NOT outperform image gradient magnitude**. In fact, simple 2D Sobel edge magnitude correlates slightly higher with empirical reconstruction error than TTA variance ($p < 0.05$).

---

## 6. Critical Scientific Analysis & Diagnostic Insights

```
┌────────────────────────────────────────────────────────────────────────┐
│ Why Does Image Gradient Match or Beat Deep TTA Uncertainty?           │
│                                                                        │
│ 1. Point Spread Function & High Frequencies:                            │
│    Reconstruction error in satellite super-resolution is highest      │
│    where spatial frequencies are highest (edges, boundaries, corners). │
│                                                                        │
│ 2. Sub-Pixel Shift Penalty:                                            │
│    Even with optimal integer shift compensation (median ~1 px),       │
│    residual sub-pixel shift (~0.3-0.5 px) causes residual spikes      │
│    precisely on high-contrast edges.                                  │
│                                                                        │
│ 3. What TTA Actually Learns:                                           │
│    The ESRGAN deep feature layers produce rotational ambiguity at     │
│    the exact same high-frequency edge transitions.                    │
│                                                                        │
│ 4. Scientific Consequence:                                            │
│    TTA uncertainty acts primarily as a high-frequency edge detector,  │
│    explaining why Sobel gradient achieves comparable AURG (0.0122     │
│    vs 0.0125) with zero neural network overhead.                     │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.1. Distinction Between Uncertainty Ranking and Calibration
- **Uncertainty Ranking**: The empirical ranking of pixels from most uncertain to least uncertain. EXP-014 demonstrates that $D_4$ TTA achieves meaningful ranking capability ($\rho \approx 0.19$, $\text{AURG} > 0$).
- **Predictive Uncertainty**: Providing an interval $[\hat{y} - 2\sigma, \hat{y} + 2\sigma]$ that claims to capture the ground truth with $95\%$ probability.
- **Probabilistic Calibration**: Demonstrating that empirical error coverage matches theoretical confidence levels.
- **Formal Status**: **$D_4$ TTA variance is NOT calibrated predictive uncertainty**. It has not been scaled via temperature scaling, Platt scaling, or conformal prediction, and does not provide valid probability intervals. It must be described solely as an **uncalibrated error-ranking heuristic**.

### 6.2. Non-Claim on Building-Level Accuracy
- As established in EXP-013, **zero semantic building footprints, masks, or vector labels exist locally**.
- High uncertainty along a structural edge indicates that the deep model produced differing textures under rotation; it does **not** prove whether a building is missing, hallucinated, or geographically displaced.
- Any academic presentation or publication must strictly state that building-level accuracy cannot be verified without external vector ground truth.

---

## 7. Deliverables and Inventory

All artifacts for EXP-014 have been saved under:
[`results/satlas_sr_experiment_014_tta_uncertainty/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/)

1. [`split_manifest.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json): Frozen deterministic split manifest (107 dev, 107 locked holdout, 42 quarantine, $>3.3\text{ km}$ buffer).
2. [`run_dev_evaluation.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/run_dev_evaluation.py): Reproducible evaluation runner script.
3. [`per_tile_uncertainty_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/per_tile_uncertainty_metrics.csv): Per-tile records for all 107 development tiles.
4. [`summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/summary.json): Complete machine-readable distribution statistics and hypothesis test results.
5. [`maps/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/maps/): 107 compressed `.npz` files containing spatial maps of $U_{\text{TTA}}$, variance, and compensated/uncompensated error.
6. [`run_command_record.txt`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/run_command_record.txt): Transparent execution trace.
7. [`EXP014_TTA_UNCERTAINTY_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_014_tta_uncertainty/EXP014_TTA_UNCERTAINTY_REPORT.md): This report.

---

## 8. Recommendations Regarding Next Experiments

### Does EXP-014 Justify Proceeding to the Locked Holdout?
**NO — Not yet.**
Evaluating the locked holdout at this stage is **premature and not recommended** for the following scientific reasons:
1. $D_4$ TTA uncertainty does not yet outperform the simple image gradient baseline ($\rho = 0.1910$ vs $0.2220$). Evaluating the holdout now would simply consume the pristine test partition to confirm that TTA is slightly inferior to a Sobel filter.
2. **Path Forward on Development Set**: The development set should first be used to explore whether a **hybrid estimator** ($U_{\text{hybrid}} = \alpha U_{\text{TTA}} + \beta U_{\text{input}}$) or a **post-hoc conformalized calibration** can meaningfully outperform the gradient baseline.
3. The holdout partition remains safely locked, uncompromised, and available for final independent validation once a competitive method is developed.
