# EXP-013: Feasibility Assessment of Candidate Uncertainty Estimation Methods

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Python Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Audit Scope**: Task D — Theoretical, Architectural, and Empirical Feasibility Assessment of 4 Uncertainty Approaches  
**Audit Status**: COMPLETED  
**Date**: 2026-10-23  

---

## 1. Executive Summary

This deliverable evaluates four candidate uncertainty estimation methods to address the central research question:

> *"Can uncertainty estimates identify reconstructed satellite-image details that are likely to contain substantial errors, especially around buildings and other fine structures?"*

The assessment was conducted strictly read-only, drawing upon the architecture of Satlas-SR ESRGAN (`ssr/models/esrgan.py`), the frozen evaluation pipeline, the EXP-011 frame-selection findings, and the registration diagnostics from EXP-004 through EXP-012.

### Summary Verdict by Method:

| Method | Candidate Name | Locally Implementable Today? | Local Evaluability with Existing Data | Requires Additional Training? | Spatial Registration Prerequisite? | Research Suitability & Readiness |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Method 1** | **Input-Selection Sensitivity** | **YES** | **High** | NO | Low (Internal variance) | **Viable Proxy**: Measures input-draw sensitivity ($s_\Delta \approx 0.89\text{ dB}$). Not calibrated uncertainty on its own. |
| **Method 2** | **Test-Time Augmentation (TTA)** | **YES** | **High** | NO | Low (Internal variance) | **Highly Feasible**: Discrete $D_4$ group (8 transforms) requires 0 interpolation. Measures model rotational equivariance variance. |
| **Method 3** | **Model Ensemble** | **NO** | **Zero** | **YES** | High | **Infeasible Locally**: Only 1 checkpoint exists (`esrgan_8S2.pth`). Cannot be evaluated without external training. |
| **Method 4** | **Error Calibration** | **PARTIAL** | **Moderate** (on 214 tiles) | NO | **Strictly Mandatory** | **Evaluative Framework**: Can evaluate if Methods 1/2 correlate with NAIP residuals, but severely confounded by temporal gaps on building boundaries. |

---

## 2. Detailed Method-by-Method Assessment

### 2.1. Method 1: Input-Selection Sensitivity

#### Concept & Mechanism
The Satlas-SR dataloader ingests 8 Sentinel-2 temporal frames out of an available pool of 32 observations. By drawing multiple distinct 8-frame subsets (e.g., using the strategies developed in EXP-011: S0, S1, S3, S4, S5), the model generates $K$ independent super-resolved predictions $\hat{Y}_k \in \mathbb{R}^{128 \times 128 \times 3}$. The pixel-wise variance $\sigma_{\text{input}}^2(x, y) = \frac{1}{K-1}\sum_{k=1}^K (\hat{Y}_k(x, y) - \bar{Y}(x, y))^2$ serves as a spatial uncertainty map.

#### Empirical Evidence from EXP-011 & EXP-012
- In EXP-011 and EXP-012, 7 strategies were evaluated across all 256 tiles. While the *mean* cPSNR was invariant across strategies ($\Delta < 0.03\text{ dB}$, $p > 0.35$), the **per-tile sample standard deviation of difference was $s_\Delta \approx 0.89\text{ dB}$**.
- Certain tiles exhibited differences exceeding $\pm 2.5\text{ dB}$ when different observation subsets were selected, driven by transient cloud contamination, seasonal illumination shifts, or atmospheric haze in specific frames.

#### Rigorous Evidentiary Caveat
> [!IMPORTANT]
> **Prediction variation measures sensitivity to input selection; it is NOT automatically calibrated uncertainty.**  
> A high variance indicates that different temporal frames produce divergent outputs (e.g., due to a cloud passing in frame 12). However, if *all* 32 frames share the same atmospheric haze, or if a building was constructed after the Sentinel-2 passes, the model may produce a confidently wrong prediction with near-zero variance. Input variance measures *epistemic ambiguity of the temporal sample*, not total predictive uncertainty.

#### Feasibility Matrix
- **Required Inputs & Artifacts**: The existing 32-frame Sentinel-2 stacks and the single frozen checkpoint (`experiments/esrgan_8S2.pth`).
- **Expected Computational Cost**: $K \times$ forward passes. For $K=5$ strategies on 256 tiles, runtime is $\sim 5 \times 12\text{ s} \approx 60\text{ seconds}$ on an RTX 2000 Ada GPU.
- **Advantages**: Requires zero new code dependencies; captures temporal instability and atmospheric noise directly.
- **Limitations**: Invariant to systematic model hallucinations that are consistent across frames; does not quantify epistemic model weight uncertainty.
- **Registration Prerequisite**: Low for computing internal variance (frames are pre-co-registered within the S2 stack); high if correlating variance against NAIP error.
- **Evidence Required Before Calling Calibrated**: Must prove through monotonic sparsification error curves (AUSE) that masking high-variance pixels monotonically reduces ground-truth reconstruction error.

---

### 2.2. Method 2: Test-Time Augmentation (TTA)

#### Concept & Mechanism
Test-Time Augmentation evaluates the model under a group of geometric transformations $\mathcal{T} = \{T_1, \dots, T_M\}$. For each transform $T_m$, the Sentinel-2 input tensor $X$ is transformed: $X_m = T_m(X)$, passed through the frozen ESRGAN model: $\hat{Y}_m = f_\theta(X_m)$, and mapped back to the canonical frame using the inverse transform: $\tilde{Y}_m = T_m^{-1}(\hat{Y}_m)$.

Uncertainty is quantified as the disagreement across augmented predictions:
$$\sigma_{\text{TTA}}^2(x, y) = \frac{1}{M}\sum_{m=1}^M \|\tilde{Y}_m(x, y) - \bar{Y}(x, y)\|^2$$

#### Mathematical Group & Interpolation Pitfalls
To preserve scientific validity, the transformation group must be strictly limited to the **dihedral group $D_4$ of the square** (order 8):
$$\mathcal{T}_{D_4} = \{\text{rot0}, \text{rot90}, \text{rot180}, \text{rot270}, \text{flip\_h}, \text{flip\_v}, \text{transp}, \text{transverse}\}$$

- **Why Discrete $D_4$ is Essential**:
  - Rotations by $90^\circ, 180^\circ, 270^\circ$ and axis reflections involve purely integer index permutations (stride reversal and axis transposition).
  - They require **exact zero spatial interpolation** (no bilinear or bicubic filtering).
  - The inverse mapping $T_m^{-1}$ is mathematically exact: $T_m^{-1}(T_m(I)) \equiv I$ at integer pixel coordinates.
- **Pitfall of Continuous Transformations**:
  - Continuous rotations (e.g., $15^\circ$, $30^\circ$) or scaling/affine warps require sub-pixel resampling.
  - Forward resampling introduces low-pass blurring; inverse resampling introduces secondary interpolation blur.
  - High variance in continuous TTA would primarily reflect resampling interpolation error rather than true model uncertainty!

#### Feasibility Matrix
- **Required Inputs & Artifacts**: Existing Sentinel-2 inputs and single checkpoint.
- **Expected Computational Cost**: $8\times$ forward passes. For 256 tiles, $\sim 8 \times 12\text{ s} \approx 96\text{ seconds}$ on an RTX 2000 Ada GPU.
- **Advantages**: Cleanly isolates the model's failure to maintain rotational equivariance. Deep convolutional networks that lack exact rotation equivariance produce varying high-frequency textures for different orientations; where the model is confident in a sharp building edge, the edge should align identically under all 8 transformations.
- **Limitations**: Measures rotational sensitivity; does not reflect missing sensor information or atmospheric distortion.
- **Registration Prerequisite**: Zero for computing the uncertainty map itself; high if correlating against NAIP.
- **Evidence Required Before Calling Calibrated**: Demonstrating that pixels with high TTA variance correspond to false hallucinations or distorted building edges.

---

### 2.3. Method 3: Deep Model Ensemble

#### Concept & Mechanism
A deep ensemble trains $E \ge 5$ independent instances of the ESRGAN model with different random weight initializations, data shuffling orders, or architectures. The ensemble variance $\sigma_{\text{ensemble}}^2(x, y) = \frac{1}{E-1}\sum_{e=1}^E (f_{\theta_e}(X) - \bar{f}(X))^2$ provides the gold-standard frequentist estimate of epistemic model uncertainty.

#### Local Checkpoint Inventory & Hard Constraint
A thorough filesystem scan of the repository was conducted:
- Local `.pth` checkpoints found: Exactly **1 file** (`experiments/esrgan_8S2.pth`, SHA256 `6e0767cb...`, 66.7 MB).
- Alternative checkpoints or model weights: **Zero**.
- Training scripts: `ssr/train.py` exists, but the audit mandate explicitly prohibits training or fine-tuning models.

#### Evidentiary Ruling
> [!CAUTION]
> **Repeated inference using the identical checkpoint (`esrgan_8S2.pth`) with different random seeds does NOT constitute a model ensemble.**  
> ESRGAN inference is completely deterministic (no dropout layers are active at evaluation time). Evaluating the same checkpoint repeatedly yields identical weights and identically zero variance ($\Delta = 0.000000$).
> 
> Therefore, **Method 3 is strictly INFEASIBLE within the local repository without external model training**.

---

### 2.4. Method 4: Error Calibration

#### Concept & Mechanism
Error calibration is not an uncertainty generation method itself, but the **formal evaluative framework** required to test whether candidate uncertainty scores (from Method 1, Method 2, or their combination) reliably predict true reconstruction error.

Given a pixel-level uncertainty map $U(x, y) \in [0, 1]$ and an empirical pixel reconstruction error map $E(x, y) = \|\hat{Y}(x, y) - Y_{\text{ref}}(x, y)\|_1$:
1. **Sparsification Analysis (Oracle vs. Uncertainty)**:
   - Rank-order all pixels in descending order of uncertainty $U(x, y)$.
   - Progressively remove the top $p\%$ most uncertain pixels ($p \in [0, 99\%]$) and compute the reconstruction error (e.g., cPSNR or MAE) on the remaining $(100 - p)\%$ pixels.
   - *Oracle Sparsification Curve*: Pixels removed in descending order of true error $E(x, y)$.
   - *Random Sparsification Curve*: Pixels removed uniformly at random (flat line).
   - *Metric*: **Area Under Sparsification Error (AUSE)** and **Area Under Random Gain (AURG)**. A valid uncertainty score must achieve $\text{AUSE} \ll \text{AUSE}_{\text{random}}$.
2. **Expected Calibration Error for Regression (ECE-R)**:
   - Group predicted uncertainties into confidence bins and measure whether empirical error coverage matches theoretical confidence intervals.

#### Confounders and Prerequisites in the Satlas-SR Dataset
Testing error calibration against the local NAIP reference images faces severe physical confounders:
1. **Sub-Pixel Spatial Misalignment**: If the SR image and NAIP reference are shifted by 1.74 pixels, high error $E(x, y)$ will appear at every sharp edge (building boundaries, roads) purely due to registration offset. An uncertainty estimator that flags true model errors would appear uncalibrated against shifted reference data.
2. **Temporal Mismatch Confounding**: Changes between 2019 NAIP and Sentinel-2 (demolitions, new roofs, seasonal vegetation) produce massive residual errors $E(x, y)$ that are not super-resolution errors.
3. **Quarantine Requirement**: Calibration analysis must be restricted strictly to the **214 reliable tiles**, excluding the 42 flagged tiles (clouds, water, $>15\text{ px}$ shift).

---

## 3. Comparative Synthesis and Feasibility Matrix

| Method Dimension | Method 1: Input Sensitivity | Method 2: Test-Time Augmentation | Method 3: Model Ensemble | Method 4: Error Calibration |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematical Basis** | Temporal sample variance $\sigma_{\text{input}}^2$ | Rotational equivariance variance $\sigma_{\text{TTA}}^2$ | Parameter distribution variance $\sigma_{\text{param}}^2$ | Correlation of $U(x, y)$ with true error $E(x, y)$ |
| **Data Requirements** | 32-frame S2 stack (available) | Single 8-frame S2 input (available) | Multiple trained checkpoints (**absent**) | Matched SR prediction and NAIP reference |
| **Code Modification Required** | Minimal (wrapper over S0..S6) | Minimal (D4 transform wrapper) | Major (retraining required) | Standard evaluation scripts (AUSE / ECE) |
| **GPU Inference Cost** | $\approx 5\times$ forward passes ($\sim 60\text{ s}$) | $\approx 8\times$ forward passes ($\sim 96\text{ s}$) | $E\times$ forward passes | Post-processing only ($< 10\text{ s}$) |
| **Susceptibility to Registration Error** | Low (internal) | Low (internal) | Low (internal) | **Extremely High** |
| **Susceptibility to Temporal Confounders** | High (captures seasonal frame differences) | Low (operates on fixed frame set) | Dependent on training data | **Extremely High** |
| **Scientific Verdict** | **Recommended for input uncertainty** | **Recommended for structural uncertainty** | **Rejected (Infeasible)** | **Mandatory Evaluation Metric** |

---

## 4. Methodological Recommendation

Rather than picking a single method solely based on ease of execution:
1. **Method 3 is ruled out** due to lack of local ensemble checkpoints.
2. **Method 2 (Test-Time Augmentation over $D_4$)** is the most rigorous method for evaluating **structural edge uncertainty around buildings**, because discrete $D_4$ preserves pixel coordinates exactly without introducing interpolation blur, directly probing the model's structural consistency.
3. **A Hybrid Formulation ($U_{\text{hybrid}} = \alpha U_{\text{TTA}} + \beta U_{\text{input}}$)** provides the most comprehensive representation, capturing both model geometric sensitivity (TTA) and temporal observation instability (input selection).
4. **Method 4** must serve as the validation protocol on the 214 reliable tiles, utilizing translation-compensated residuals to prevent registration jitter from masquerading as calibration failure.
