# EXP-013: Scientifically Defensible Next Experiment Protocol

**Protocol Identifier**: EXP-014-PROTOCOL  
**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Protocol Status**: PROPOSED PROTOCOL (Strictly Read-Only Planning — No Experiments Executed)  
**Date**: 2026-10-23  

---

## 1. Executive Summary & Research Alignment

### 1.1. Core Research Question
> *"Can uncertainty estimates identify reconstructed satellite-image details that are likely to contain substantial errors, especially around buildings and other fine structures?"*

### 1.2. Protocol Purpose & Strict Boundary
This protocol defines a rigorous, falsifiable experimental procedure to test whether pixel-level uncertainty metrics derived from the frozen Satlas-SR ESRGAN baseline predict empirical reconstruction errors. 

**This document specifies a future protocol only. No inference, training, threshold tuning, or external downloads are authorized or performed during EXP-013.**

---

## 2. Prespecified Experimental Baselines & Metrics

### 2.1. Candidate Uncertainty Methods & Baselines

To prove that a proposed uncertainty estimator has genuine scientific value, it must be benchmarked against both naive and heuristic baselines:

1. **Baseline 0: Uniform Random Uncertainty ($U_{\text{random}}$)**:
   - Pixels assigned random values $U(x, y) \sim \mathcal{U}(0, 1)$. Represents uninformative chance.
2. **Baseline 1: High-Frequency Texture Complexity ($U_{\text{texture}}$)**:
   - Local gradient magnitude of the bicubic-upsampled low-resolution Sentinel-2 input (Sobel filter magnitude). Tests whether uncertainty simply flags busy/textured regions regardless of model confidence.
3. **Candidate Method 1: Input-Selection Sensitivity ($U_{\text{input}}$)**:
   - Pixel variance across $K=5$ deterministic observation subsets (S0, S1, S3, S4, S5 from EXP-011).
4. **Candidate Method 2: Test-Time Augmentation ($U_{\text{TTA}}$)**:
   - Pixel variance across the discrete $D_4$ dihedral transformation group (8 exact transforms, 0 interpolation).
5. **Candidate Method 3: Hybrid Uncertainty ($U_{\text{hybrid}}$)**:
   - Linear combination: $U_{\text{hybrid}}(x, y) = \alpha \cdot \tilde{U}_{\text{TTA}}(x, y) + (1 - \alpha) \cdot \tilde{U}_{\text{input}}(x, y)$, where weights are tuned strictly on the development split.

### 2.2. Prespecified Evaluation Metrics

1. **Area Under Sparsification Error (AUSE $\downarrow$)**:
   - The area between the empirical sparsification error curve and the theoretical oracle sparsification curve:
     $$\text{AUSE} = \int_{0}^{0.95} (\text{Error}(p) - \text{Error}_{\text{oracle}}(p)) \, dp$$
   - Smaller AUSE indicates superior uncertainty ranking.
2. **Area Under Random Gain (AURG $\uparrow$)**:
   - The area between the empirical sparsification curve and the random baseline curve:
     $$\text{AURG} = \int_{0}^{0.95} (\text{Error}_{\text{random}}(p) - \text{Error}(p)) \, dp$$
   - A valid uncertainty metric must achieve $\text{AURG} > 0$.
3. **Spearman Rank Correlation ($\rho \uparrow$)**:
   - Non-parametric rank correlation between predicted uncertainty $U(x, y)$ and empirical absolute residual error $E_{\text{reg}}(x, y) = |\hat{Y}(x, y) - Y_{\text{ref}}(x + \Delta x, y + \Delta y)|$.
4. **Expected Calibration Error for Regression (ECE-R $\downarrow$)**:
   - Measures whether the empirical coverage of uncertainty confidence intervals matches theoretical expectations.

---

## 3. Five-Stage Experimental Protocol

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: REFERENCE VALIDATION                        │
│ Filter 214 reliable tiles (exclude 42 flagged); compute cPSNR shifts.  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: UNCERTAINTY PROTOTYPE                       │
│ Compute discrete D4 TTA and 5-strategy Input Sensitivity maps.         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: RELIABILITY EVALUATION                      │
│ Sparsification curves, AUSE, AURG, Spearman rho against NAIP residuals.│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: STRUCTURAL / EDGE ANALYSIS                  │
│ High-gradient edge residuals (proxy for building boundaries).          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 5: INDEPENDENT HOLDOUT VALIDATION              │
│ 50/50 split (107 dev / 107 test); evaluate once on locked test set.     │
└────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Reference Validation and Sample Quarantine
- **Objective**: Prevent known spatial registration failures and cloud artifacts from corrupting calibration metrics.
- **Protocol**:
  1. Restrict evaluation strictly to the **214 reliable tiles** identified in `results/satlas_sr_spatial_audit/registration_validation.csv` (83.6% of the dataset).
  2. Permanently quarantine the **42 unreliable tiles** (37 low-correlation scenes with NCC $< 0.25$ and 5 severe outliers with shift $> 15\text{ px}$).
  3. For each of the 214 tiles, apply the rigid integer shift $[\Delta x^*, \Delta y^*] \in [-8, 8] \times [-8, 8]$ determined by cPSNR optimization before computing pixel-wise residual errors:
     $$E_{\text{reg}}(x, y) = \|\hat{Y}(x, y) - Y_{\text{ref}}(x + \Delta x^*, y + \Delta y^*)\|_1$$
     This guarantees that evaluated errors reflect reconstruction quality rather than bulk 1–2 pixel translation jitter.

### Stage 2: Uncertainty Map Generation
- **Objective**: Generate test-time uncertainty maps without modifying the trained model weights.
- **Protocol**:
  1. **TTA Pipeline**: Pass each Sentinel-2 input through the 8 transformations of the discrete dihedral group $D_4$ using pure tensor dimension transposition and flipping. Invert each prediction to canonical orientation and compute pixel-wise variance across the 8 passes: $\sigma^2_{\text{TTA}}(x, y)$.
  2. **Input Selection Pipeline**: Pass 5 deterministic 8-frame selections (S0, S1, S3, S4, S5 from EXP-011) through the model and compute pixel-wise variance: $\sigma^2_{\text{input}}(x, y)$.
  3. Normalize both maps to $[0, 1]$ per tile using 1st and 99th percentile min-max scaling to prevent outlier skewing.

### Stage 3: Reliability & Error Calibration Evaluation
- **Objective**: Quantify whether uncertainty maps predict true reconstruction errors.
- **Protocol**:
  1. For each tile, sort all interior pixels (excluding a 4-pixel border to match cPSNR evaluation conventions) by uncertainty value.
  2. Generate sparsification curves by progressively masking out pixels from $p = 0\%$ to $p = 95\%$ in increments of $1\%$.
  3. Recompute Mean Absolute Error (MAE) and cPSNR on the remaining unmasked pixels at each step.
  4. Integrate the sparsification curve to compute AUSE and AURG against the random baseline and the theoretical oracle.

### Stage 4: Structural & Edge-Proxy Analysis (Bridge to Buildings)
- **Objective**: Assess uncertainty performance specifically around sharp, fine structural boundaries.
- **Protocol**:
  1. Acknowledge and document that **zero semantic building masks exist locally**.
  2. As a defensible proxy for building edges, apply a Sobel edge filter to the NAIP reference image to create a binary high-gradient edge mask $M_{\text{edge}}$ (top 15% gradient magnitude pixels).
  3. Compute stratified sparsification metrics specifically for edge pixels ($M_{\text{edge}} = 1$) vs background pixels ($M_{\text{edge}} = 0$).
  4. *Precondition for True Building Evaluation*: Formally record that before any claims of "building-level mapping reliability" can be published, external vector building footprints (e.g., Microsoft US Building Footprints) must be acquired and co-registered.

### Stage 5: Independent Split and Holdout Evaluation
- **Objective**: Prevent data leakage, selective thresholding, or overfitting.
- **Protocol**:
  1. Partition the 214 reliable tiles into two stratified subsets based on geographic latitude and scene brightness:
     - **Development / Tuning Split ($N=107$ tiles)**: Used for hyperparameter tuning (e.g., finding optimal weight $\alpha$ for hybrid uncertainty, setting sparsification cutoff thresholds).
     - **Locked Holdout Test Split ($N=107$ tiles)**: Locked and untouched during development. Evaluated strictly once with frozen parameters.
  2. Final claims, tables, and hypothesis tests must be reported exclusively on the locked holdout split.

---

## 4. Confounder Identification & Risk Mitigation

| Confounder | Physical Cause | Potential Impact on Calibration | Mitigation Protocol |
| :--- | :--- | :--- | :--- |
| **Sub-pixel registration jitter** | Residual satellite vs airborne positioning offset ($~1.7\text{ px}$). | Large false error spikes along every true edge; penalizes sharp, accurate edges. | Compensate with cPSNR optimal translation vector $[\Delta x^*, \Delta y^*]$ before calculating residuals. |
| **Temporal land-cover change** | 2019 NAIP survey vs multi-year S2 observations (new buildings, cleared land). | Massive residual errors where model predicts current reality but reference depicts old state. | Quarantine low-correlation tiles; conduct stratified analysis on high-confidence invariant textures. |
| **Solar shadow variation** | Different sun angles between aerial flight and satellite pass. | Shadow boundaries shift by 5–10 meters, creating spurious edge residuals. | Evaluate using luminance-normalized residuals or structural SSIM error maps rather than raw RGB differences. |
| **Atmospheric haze / cirrus** | Thin cloud contamination in one or more Sentinel-2 frames. | Inflates input sensitivity variance on cloud-covered scenes. | Compare input sensitivity against TTA, which operates on fixed frame sets. |

---

## 5. Predefined Falsifiable Success Criteria

Before claiming that uncertainty estimates are scientifically viable, the proposed experiment must satisfy the following predefined numerical criteria on the **locked holdout test split ($N=107$)**:

1. **Statistical Superiority over Random Baseline**:
   - Mean AURG for $U_{\text{TTA}}$ and $U_{\text{hybrid}}$ must be strictly greater than zero with paired Wilcoxon signed-rank test $p < 0.001$.
2. **Meaningful Error Correlation**:
   - The Spearman rank correlation between uncertainty and translation-compensated residual error must satisfy:
     $$\text{Median } \rho(U, E_{\text{reg}}) \ge 0.30$$
3. **Monotonic Error Sparsification**:
   - Progressively masking the top 20% most uncertain pixels must reduce the residual MAE by at least **$15\%$** relative to unmasked MAE across the holdout set.
4. **Edge Stratification Criterion**:
   - On high-gradient edge pixels ($M_{\text{edge}} = 1$), the uncertainty score must exhibit higher rank correlation ($\rho_{\text{edge}} > \rho_{\text{background}}$), demonstrating sensitivity to fine structure.

If these criteria are not met, the hypothesis that test-time uncertainty identifies reconstruction errors must be formally rejected or reformulated.
