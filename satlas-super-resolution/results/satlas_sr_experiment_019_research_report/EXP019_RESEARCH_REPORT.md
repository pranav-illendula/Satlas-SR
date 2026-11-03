# Evaluating Hybrid Uncertainty Estimation for Multi-Frame Satellite Super-Resolution: A Frozen Independent Holdout Study

**Experiment Code:** EXP-019  
**Repository:** `satlas-super-resolution`  
**Git Commit Hash:** `3e67973f6746bedbd91cc032b4888574980e1daf`  
**Execution Environment:** Windows (x64), NVIDIA RTX 2000 Ada Generation, Python 3.10.11 virtual environment  
**Deliverables Directory:** `results/satlas_sr_experiment_019_research_report/`  
**Primary Source Artifacts:** EXP-011 through EXP-018  
**Date:** 2026-10-23  

---

## Abstract

Deep learning super-resolution models operating on multi-temporal satellite imagery can hallucinate fine spatial details or amplify atmospheric artifacts. Determining which reconstructed pixels are trustworthy is essential for high-stakes geospatial applications. This report synthesizes a multi-phase empirical investigation into pixel-level uncertainty estimation for the Satlas-SR ESRGAN baseline ($4\times$ upsampling from $10\text{ m}$ Sentinel-2 to $2.5\text{ m}$ NAIP-compatible resolution). 

We developed and evaluated a frozen **Hybrid Uncertainty Estimator** that combines **$D_4$ test-time augmentation (TTA) orientation sensitivity** with **Sentinel-2 temporal input-selection sensitivity** via feature standardization and convex blending ($\alpha^* = 0.19$). Following cross-validation on a 107-tile development set (EXP-015) and an independent audit (EXP-016), the estimator was frozen and evaluated **exactly once** on a locked, previously uninspected holdout partition of 107 geographic tiles (EXP-017). 

On the holdout partition, the hybrid estimator achieved a mean per-tile Spearman rank correlation against ground-truth reconstruction error of **$\bar{\rho} = 0.279966$** (95% CI: $[0.251292, 0.308639]$), compared to **$\bar{\rho} = 0.200649$** (95% CI: $[0.173331, 0.227968]$) for a standard Sobel image-gradient baseline. The primary endpoint—paired per-tile difference against the baseline—demonstrated a statistically significant advantage of **$\bar{\Delta} = +0.079316$** (95% CI: $[+0.061084, +0.097548]$, paired $t = 8.6251$, $p = 6.87 \times 10^{-14}$; Wilcoxon $W = 458.0$, $p = 4.17 \times 10^{-14}$; Cohen's $d = 0.8338$), winning on **$93$ out of $107$ holdout tiles ($86.92\%$)**. In practical sparsification analysis, discarding the 20% highest-uncertainty pixels reduced reconstruction Mean Absolute Error (MAE) by **$11.27\%$** for the hybrid estimator versus **$8.43\%$** for the baseline ($p = 2.75 \times 10^{-16}$). 

The paired treatment effect replicated within $0.0029$ correlation units of development findings ($\bar{\Delta} = +0.082205$, $89.72\%$ win rate). Multi-temporal observation sensitivity was found to drive the vast majority of the performance advantage. We distinguish statistical significance from practical error-ranking utility, present reconciled errata for earlier report tables, and articulate strict boundaries: uncertainty scores represent empirical error-ranking heuristics, not calibrated probabilities, geodetic alignments, or building-level boundary guarantees.

---

## 1. Research Question and Motivation

### 1.1. The Reliability Problem in Satellite Super-Resolution
Super-resolution (SR) of satellite imagery—specifically reconstructing fine ground features from multi-temporal low-resolution observations—is inherently ill-posed. Deep convolutional architectures such as Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) optimize perceptual realism, often synthesizing high-frequency textures that appear plausible but diverge substantially from actual ground truth. In earth observation workflows (e.g., building footprint extraction, infrastructure monitoring, disaster assessment), undetected hallucinated edges or smoothed structures introduce severe systematic errors.

### 1.2. Research Question
This project investigated a fundamental technical question:
> **Can post-hoc, test-time uncertainty estimators identify reconstructed satellite-image pixels that are likely to contain substantial reconstruction error, outperforming simple image-gradient heuristics?**

While conventional computer vision literature frequently proposes test-time augmentation (TTA) or dropout-based epistemic uncertainty, prior experiments in this project (EXP-014) revealed that $D_4$ TTA uncertainty alone achieved a modest Spearman error correlation ($\bar{\rho} = 0.1910$) that failed to outperform a simple 2D Sobel image gradient baseline ($\bar{\rho} = 0.2220$). 

This motivated the exploration of a **Hybrid Estimator** that combines spatial/geometric transformation sensitivity with **Sentinel-2 multi-temporal observation sensitivity**—leveraging the physical reality that satellite super-resolution models must fuse multiple revisits subject to variable cloud cover, atmospheric scattering, and coregistration noise.

---

## 2. Dataset, Quality Audits, and Evaluation Partitions

### 2.1. Dataset Composition
The evaluation utilized the released validation set of the Satlas-SR benchmark, consisting of:
- **Low-Resolution Inputs:** Sentinel-2 Level-1C/2A Top-Of-Atmosphere / Bottom-Of-Atmosphere reflectance tiles ($32 \times 32$ pixels, 9–18 temporal revisits per tile, 12 spectral bands reduced to RGB chunks).
- **High-Resolution Reference:** National Agriculture Imagery Program (NAIP) aerial imagery ($128 \times 128$ pixels, $3$-channel RGB, $\approx 0.5\text{–}1.0\text{ m}$ spatial resolution resampled to match $4\times$ SR).

### 2.2. Geospatial Alignment and Quality Quarantine (EXP-013)
Before defining evaluation splits, an exhaustive geospatial integrity audit was conducted in EXP-013 across all 256 tiles:
- **Corrupted Reference Exclusion:** 42 tiles were found to suffer from extreme multi-pixel registration shifts ($>8$ pixels), severe temporal discordance (e.g., snow cover in NAIP vs summer grass in Sentinel-2), or NAIP border clipping. These 42 tiles were formally quarantined and permanently excluded from all experiments.
- **Eligible Population:** Exactly **214 valid tiles** were established as the eligible benchmark population.
- **Residual Registration Policy:** Even among high-quality tiles, minor sub-pixel and integer misalignments occur between satellite ephemeris and aerial orthomosaics. To prevent registration artifacts from dominating pixel-level error evaluations, an optimal integer translation search was established:
  $$(\Delta x^*, \Delta y^*) = \arg\min_{(\Delta x, \Delta y) \in [-8, 8]^2} \frac{1}{|W|} \sum_{(u, v) \in W} \left( \hat{I}(u, v) - I_{\text{ref}}(u + \Delta x, v + \Delta y) \right)^2$$
  evaluated over an interior evaluation window $W$ ($120 \times 120$ pixels, excluding a 4-pixel outer border to match cPSNR protocols).

### 2.3. Partition Freezing and Spatial Buffering (EXP-014)
To ensure absolute independence and prevent data snooping:
- The 214 eligible tiles were split deterministically into two equal partitions of **107 development tiles** and **107 locked holdout tiles**.
- **Spatial Separation:** Nationwide spatial clustering was evaluated. Multi-tile clusters within Chebyshev distance $\le 10$ tiles ($\approx 3.3\text{ km}$) were strictly preserved within their respective partitions.
- **Holdout Locking:** The 107 holdout tiles were sealed in `results/satlas_sr_experiment_014_tta_uncertainty/split_manifest.json`. No holdout images, predictions, or reference errors were inspected or referenced during method formulation or hyperparameter tuning.

---

## 3. The Frozen Hybrid Uncertainty Estimator

### 3.1. Mathematical Formulation
The hybrid uncertainty estimator fuses two orthogonal error signals:
1. **$D_4$ Test-Time Augmentation Sensitivity ($U_{\text{tta}}$):**
   Evaluates the sensitivity of the model to discrete 2D spatial symmetries of the square lattice ($4$ rotations $\times$ $2$ horizontal reflections = $8$ symmetries in group $D_4$). For input tensor $X$, symmetry operator $T_g$, and model $f_\theta$:
   $$\hat{Y}_g = T_g^{-1}\left( f_\theta(T_g(X)) \right), \quad g \in D_4$$
   $$U_{\text{tta}}(u, v) = \sqrt{ \frac{1}{8-1} \sum_{g=1}^8 \left( \hat{Y}_g(u, v) - \bar{Y}(u, v) \right)^2 }$$
   $U_{\text{tta}}$ detects rotational and geometric ambiguity, peaking along complex directional textures.
2. **Sentinel-2 Temporal Input-Selection Sensitivity ($U_{\text{input}}$):**
   Evaluates the sensitivity of the model to the selection and temporal ordering of the 8 Sentinel-2 input frames. Using predictions across 6 diverse frame-selection strategies documented in EXP-011 ($S_0$: baseline temporal sort, $S_1$: reverse sort, $S_3$: high-contrast subset, $S_4$: median cloud clearance, $S_5$: alternating revisit, $S_6$: random seed):
   $$U_{\text{input}}(u, v) = \sqrt{ \frac{1}{6-1} \sum_{s \in \{0, 1, 3, 4, 5, 6\}} \left( \hat{Y}_s(u, v) - \bar{Y}_{\text{input}}(u, v) \right)^2 }$$
   $U_{\text{input}}$ detects atmospheric inconsistencies, shadow shifts, cloud fringe artifacts, and temporal observation noise.

### 3.2. Normalization and Convex Blending
Because $U_{\text{tta}}$ and $U_{\text{input}}$ possess different numerical scales, standardizing both features prior to combination is mathematically essential. Feature standardization parameters were derived strictly across all $107 \times 14,400 = 1,540,800$ development-set pixels:
$$Z(U_{\text{tta}}) = \frac{U_{\text{tta}} - \mu_{\text{tta}}}{\sigma_{\text{tta}}}, \quad Z(U_{\text{input}}) = \frac{U_{\text{input}} - \mu_{\text{input}}}{\sigma_{\text{input}}}$$
with frozen normalization statistics:
$$\mu_{\text{tta}} = 0.083126, \quad \sigma_{\text{tta}} = 0.051388$$
$$\mu_{\text{input}} = 0.057286, \quad \sigma_{\text{input}} = 0.041238$$

The hybrid uncertainty score is defined as:
$$U_{\text{hybrid}} = \alpha^* \cdot Z(U_{\text{tta}}) + (1 - \alpha^*) \cdot Z(U_{\text{input}})$$
In EXP-015, grid search over $\alpha \in [0.0, 1.0]$ across 5 cross-validation training folds converged uniformly to $\alpha^* \in [0.15, 0.20]$ (mean **$\alpha^* = 0.1900$**). This parameter was frozen into `preflight_protocol.json` prior to holdout evaluation.

---

## 4. Evaluation Metrics and Baselines

### 4.1. Evaluated Estimators and Baselines
All methods were evaluated on the identical $120 \times 120$ central crop under the identical integer shift $[\Delta x^*, \Delta y^*]$:
1. **Frozen Hybrid Estimator ($U_{\text{hybrid}}$):** Blended score with $\alpha^* = 0.1900$.
2. **Input-Selection Sensitivity ($U_{\text{input}}$):** Unblended multi-temporal observation variance.
3. **$D_4$ TTA Uncertainty ($U_{\text{tta}}$):** Unblended geometric transform variance.
4. **Sobel Gradient Baseline ($U_{\text{grad}}$):** Normalized 2D spatial gradient magnitude computed on the luminance of the baseline super-resolved image.
5. **Random Baseline ($U_{\text{rand}}$):** Uniform random distribution $U(0, 1)$.
6. **Constant Baseline ($U_{\text{const}}$):** Uniform constant array ($0.5$).

### 4.2. Metric Definitions
- **Primary Endpoint — Spearman Error-Ranking Correlation ($\rho$):** Per-tile Spearman rank correlation between uncertainty map $U(u, v)$ and compensated absolute reconstruction error $E_{\text{comp}}(u, v) = |\hat{I}(u, v) - I_{\text{ref, comp}}(u, v)|$.
- **Primary Comparison — Paired Difference ($\Delta \rho$):** $\Delta \rho = \rho_{\text{hybrid}} - \rho_{\text{grad}}$ evaluated per-tile.
- **Secondary Endpoint — Area Under the Random Gain Curve (AURG):**
  $$\text{AURG} = \int_{0}^{0.95} \left( \text{MAE}_{\text{random}}(p) - \text{MAE}_{\text{model}}(p) \right) dp$$
  evaluated via composite trapezoidal integration over 20 uniform fractions $p \in [0.0, 0.95]$.
- **Secondary Endpoint — Sparsification MAE Reduction:** Percentage drop in Mean Absolute Error when the top $p \in \{10\%, 20\%, 30\%\}$ highest-uncertainty pixels are discarded.

---

## 5. Development Partition Findings (EXP-015 & EXP-016 Reconciled)

The development set was evaluated under a strict 5-fold spatially clustered cross-validation protocol (EXP-015), where spatial clusters were separated by an inter-fold buffer of $\ge 12$ tiles ($>3.6\text{ km}$). The independent audit in EXP-016 and reconciliation in EXP-018 confirmed the following ground-truth results across the 107 development tiles:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   RECONCILED DEVELOPMENT OUT-OF-FOLD RESULTS (N=107)                   │
├───────────────────────────────┬───────────────────┬──────────────────┬─────────────────┤
│ Estimator                     │ Mean Spearman ρ   │ Sample SD (s)    │ 95% Param. CI   │
├───────────────────────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Hybrid Estimator (U_hybrid)   │ 0.304210          │ 0.153850         │ [0.2747, 0.3337]│
│ Input Sensitivity (U_input)   │ 0.297557          │ 0.151680         │ [0.2685, 0.3266]│
│ Sobel Gradient Baseline       │ 0.222005          │ 0.163761         │ [0.1906, 0.2534]│
│ D4 TTA Uncertainty            │ 0.190989          │ 0.165065         │ [0.1594, 0.2226]│
│ Constant Baseline             │ 0.000000          │ 0.000000         │ [0.0000, 0.0000]│
└───────────────────────────────┴───────────────────┴──────────────────┴─────────────────┘
```

### Key Development Takeaways
- **Hybrid vs. Sobel Baseline:** Paired gain of **$\bar{\Delta} = +0.082205$** ($+37.03\%$ relative gain, paired $t = 7.4107$, $p = 3.20 \times 10^{-11}$, Wilcoxon $p = 2.11 \times 10^{-15}$, Cohen's $d = 0.7164$).
- **Per-Tile Wins:** Hybrid outperformed the Sobel baseline on **$96$ out of $107$ tiles ($89.72\%$)**, with $11$ losses and $0$ ties.
- **Component Attribution:** Input sensitivity alone achieved $\bar{\rho} = 0.297557$ (paired gain $+0.075552$ over Sobel). Measuring the ratio of paired mean gains over Sobel:
  $$\frac{\bar{\Delta}_{\text{input vs Sobel}}}{\bar{\Delta}_{\text{hybrid vs Sobel}}} = \frac{+0.075552}{+0.082205} = 91.9\%$$
  indicating that the majority of the linear paired advantage over the spatial edge detector stems from multi-temporal observation variance. Fusing $D_4$ TTA provided a secondary structural increment of $\bar{\Delta} = +0.006653$ ($p = 0.00010$, winning on $72.0\%$ of development tiles).

---

## 6. Frozen Locked-Holdout Results (EXP-017 & EXP-018 Audited)

Following protocol verification in EXP-017, the 107 holdout tiles were evaluated in a single frozen execution pass ($42.54$ seconds total runtime, $0.40$ s/tile) on an NVIDIA RTX 2000 Ada Generation GPU. All metrics were independently audited and verified in EXP-018.

### 6.1. Primary Endpoint: Paired Error-Ranking Performance

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      LOCKED HOLDOUT EVALUATION RESULTS (N=107 TILES)                   │
├───────────────────────────────┬───────────────────┬──────────────────┬─────────────────┤
│ Estimator / Baseline          │ Mean Spearman ρ   │ Sample SD (s)    │ 95% Param. CI   │
├───────────────────────────────┼───────────────────┼──────────────────┼─────────────────┤
│ Frozen Hybrid (U_hybrid)      │ 0.279966          │ 0.149602         │ [0.2513, 0.3086]│
│ Input Sensitivity (U_input)   │ 0.275427          │ 0.148836         │ [0.2469, 0.3040]│
│ Sobel Gradient Baseline       │ 0.200649          │ 0.142533         │ [0.1733, 0.2280]│
│ D4 TTA Uncertainty            │ 0.169973          │ 0.151988         │ [0.1408, 0.1991]│
│ Random Uniform Baseline       │ -0.001161         │ 0.008049         │ [-0.0027, 0.0004│
│ Constant Baseline             │ 0.000000          │ 0.000000         │ [0.0000, 0.0000]│
└───────────────────────────────┴───────────────────┴──────────────────┴─────────────────┘
```

### 6.2. Paired Hypothesis Tests and Effect Sizes vs. Sobel Baseline

| Comparison | Mean Diff ($\bar{\Delta}$) | SD ($s_{\Delta}$) | 95% Parametric CI | Paired $t$-stat | $t$-test $p$-value | Wilcoxon $W$ | Wilcoxon $p$-value | Cohen's $d$ | Win Rate (%) | Wins / Losses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid vs. Sobel** (Primary) | **+0.079316** | 0.095124 | **[+0.061084, +0.097548]** | **8.6251** | **$6.87 \times 10^{-14}$** | **458.0** | **$4.17 \times 10^{-14}$** | **0.8338** | **86.92%** | **93 / 14** |
| Input vs. Sobel | +0.074778 | 0.104309 | [+0.054785, +0.094770] | 7.4155 | $3.13 \times 10^{-11}$ | 625.0 | $1.97 \times 10^{-12}$ | 0.7169 | 85.05% | 91 / 16 |
| $D_4$ TTA vs. Sobel | -0.030676 | 0.128077 | [-0.055224, -0.006128] | -2.4775 | $0.0148$ | 1750.0 | $0.00040$ | -0.2395 | 31.78% | 34 / 73 |
| **Hybrid vs. Input** | **+0.004538** | 0.034357 | [-0.002047, +0.011123] | 1.3664 | $0.1747$ | **1783.0** | **$0.000587$** | 0.1321 | **66.36%** | **71 / 36** |
| Hybrid vs. $D_4$ TTA | +0.109992 | 0.129527 | [+0.085166, +0.134818] | 8.7840 | $3.03 \times 10^{-14}$ | 531.0 | $2.32 \times 10^{-13}$ | 0.8492 | 85.98% | 92 / 15 |

---

## 7. Statistical Analysis & Cautionary Diagnosis

### 7.1. Treatment Effect Generalization
The primary treatment effect replicated with high precision between development and holdout:
- Development Paired Mean Gain: **$\bar{\Delta} = +0.082205$** ($+37.03\%$ relative gain, $89.72\%$ win rate).
- Holdout Paired Mean Gain: **$\bar{\Delta} = +0.079316$** ($+39.53\%$ relative gain, $86.92\%$ win rate).
- Generalization Discrepancy: **$0.002889$ correlation units**. Both partitions experienced a minor baseline drop of $\approx 0.022$ correlation units on raw means (reflecting natural geographic variation across non-overlapping scenes), but the paired advantage over the spatial edge detector remained constant.

### 7.2. Critical Statistical Caution: Hybrid vs. Input-Selection Sensitivity
A key methodological question is whether combining $D_4$ TTA with Input Sensitivity provides a statistically defensible advantage over Input Sensitivity alone. On the holdout partition:
- **Parametric Paired $t$-test:** $t = 1.3664$, **$p = 0.174703$** (fails to reject null at $\alpha = 0.05$).
- **Non-Parametric Wilcoxon Signed-Rank Test:** $W = 1783.0$, **$p = 0.000587$** (strongly rejects null, $p < 10^{-3}$).
- **Per-Tile Directional Shift:** Hybrid outperformed Input Sensitivity on **$71$ of $107$ tiles ($66.36\%$)**, a nearly $2:1$ win ratio.

#### Distributional Diagnosis:
The difference array $\Delta_i = \rho_{\text{hybrid}, i} - \rho_{\text{input}, i}$ is severely non-normal:
- **Sample Skewness:** **$-4.6405$** (heavy negative skew).
- **Sample Kurtosis:** **$36.1496$** (extreme leptokurtic distribution).
- **Influential Outlier:** A single tile (`32822_54029`) exhibited an extreme negative difference of **$\Delta = -0.269097$**. (The next most negative tile was $-0.052$).

#### Interpretation and Test Assumptions:
- **Why the $t$-test Fails:** The extreme negative outlier acts as a high-leverage point. It depresses the sample mean from $+0.0072$ (the median) down to $+0.0045$, while inflating the sample standard deviation by $68\%$ (to $s_{\Delta} = 0.03436$). The inflated denominator reduces the $t$-statistic to $1.37$.
- **Wilcoxon Assumptions:** The Wilcoxon signed-rank test assumes that differences are independent and symmetric under the null hypothesis. While it bounds the numerical leverage of the $-0.2691$ outlier by transforming values to ranks $1 \dots 107$, its validity strictly relies on the assumption of symmetric difference distributions under $H_0$. 
- **Methodological Caution:** It is scientifically inaccurate to declare that the Hybrid estimator unconditionally outperforms Input Sensitivity across all statistical criteria. Under strict parametric testing, the difference is not statistically significant. However, non-parametric rank testing and sign testing ($71$ wins vs $36$ losses, binomial $p = 0.0009$) demonstrate a genuine directional advantage across two-thirds of scenes. Researchers must report both tests transparently.

### 7.3. Attribution of Performance Gain
On the holdout partition, Input Sensitivity alone achieves a paired gain of $+0.074778$ over Sobel, while the Hybrid achieves $+0.079316$. Defining attribution as the ratio of paired mean improvements:
$$\frac{\bar{\Delta}_{\text{input vs Sobel}}}{\bar{\Delta}_{\text{hybrid vs Sobel}}} = \frac{+0.074778}{+0.079316} = 94.28\%$$
While this ratio demonstrates that temporal observation sensitivity is the overwhelmingly dominant driver of error correlation, we caution that Spearman rank correlation is a non-linear statistic; linear variance attribution is an empirical approximation rather than an orthogonal variance decomposition.

---

## 8. Practical Sparsification & Downstream Utility

Evaluating rank correlation alone does not prove downstream utility. To determine whether these estimators provide operational value, we analyzed pixel error retention across progressive sparsification fractions $p \in [0.0, 0.95]$.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     HOLDOUT ERROR SPARSIFICATION PERFORMANCE (N=107)                   │
├───────────────────────────────┬───────────────────┬──────────────────┬─────────────────┤
│ Estimator / Metric            │ Hybrid Estimator  │ Sobel Baseline   │ Paired Gain (Δ) │
├───────────────────────────────┼───────────────────┼──────────────────┼─────────────────┤
│ MAE Reduction @ 10% Discard   │ 6.71% ± 3.70%     │ 5.12% ± 3.20%    │ +1.59% (p<10⁻¹²)│
│ MAE Reduction @ 20% Discard   │ 11.27% ± 6.43%    │ 8.43% ± 5.77%    │ +2.84% (p<10⁻¹⁵)│
│ MAE Reduction @ 30% Discard   │ 15.08% ± 8.95%    │ 11.16% ± 8.10%   │ +3.92% (p<10⁻¹⁶)│
│ AURG (Area Under Random Gain) │ 0.01593 ± 0.00998 │ 0.01124 ± 0.00893│ +0.00469 (p<10⁻¹³)
└───────────────────────────────┴───────────────────┴──────────────────┴─────────────────┘
```

### Operational Interpretation:
- **20% Sparsification Benchmark:** Discarding the 20% highest-uncertainty pixels identified by the Hybrid estimator achieves an **$11.27\%$ reduction in reconstruction MAE**, winning over the Sobel baseline on **$98$ out of $107$ holdout tiles ($91.59\%$, Cohen's $d = 0.9371$)**.
- **Practical Significance:** While the gain over Sobel ($+2.84\%$ absolute MAE reduction) is highly statistically significant, the remaining $80\%$ of pixels still retain substantial absolute error ($\approx 0.051$ reflectance units). Hybrid uncertainty acts as an effective screening filter for gross reconstruction failures, but does not render the retained output error-free.

---

## 9. Scientific Guardrails, Limitations, and Threats to Validity

To prevent over-claiming and ensure rigorous interpretation, the following technical limitations are formally stated:

1. **Uncertainty as Error-Ranking Heuristic, Not Calibrated Probability:**
   The hybrid estimator produces standardized relative rank scores ($Z$-score blends), not Bayesian posterior variances or calibrated probabilistic prediction intervals. The scores cannot be interpreted as percentage probabilities of error.
2. **Integer Image-Space Registration vs. Geodetic Alignment:**
   Residual translation was compensated using optimal integer shifts $(\Delta x, \Delta y) \in [-8, 8]^2$ on the central evaluation window. While this prevents sub-pixel edge alignment errors from dominating evaluation, image-space translation does not constitute geodetic orthorectification, sensor geometry modeling, or terrain displacement correction.
3. **Absence of Building-Level Segmentation Validation:**
   This study evaluated pixel-level RGB reconstruction error against NAIP imagery. No building footprint masks, polygon boundary IoU metrics, or vector mapping accuracies were evaluated. Claims regarding "building extraction calibration" are unsubstantiated by these experiments.
4. **Sample Bounding and Geographic Representativeness:**
   The holdout partition comprises exactly **107 tiles** located within the contiguous United States, selected from the existing Satlas-SR benchmark distribution. Findings cannot be generalized to tropical regions, arid desert expanses, steep mountainous terrain, or different sensor constellations without independent empirical validation.
5. **Spatial Dependence and Degrees of Freedom:**
   Although tiles were separated across cross-validation folds using a buffer of $\ge 12$ tiles ($>3.6\text{ km}$), minor spatial auto-correlation in regional landscape types across the 107 nationwide tiles cannot be completely eliminated. Even under conservative adjustments to effective degrees of freedom ($df=102$), the primary effect remains statistically significant ($p < 10^{-11}$).
6. **Metric Specificity:**
   Error ranking was assessed using Spearman rank correlation and MAE sparsification. These metrics assess monotonic error association; they do not measure variance calibration (e.g., negative log-likelihood or continuous ranked probability scores).

---

## 10. Conclusion

This project rigorously investigated whether test-time uncertainty estimation can identify unreliable super-resolved satellite imagery pixels. The empirical findings across both development ($N=107$) and frozen locked-holdout ($N=107$) partitions support the following conclusions:

1. **Test-Time Augmentation Alone is Insufficient:** $D_4$ symmetry transforms alone underperform simple 2D spatial edge gradients ($\bar{\rho} = 0.1700$ vs $0.2006$), confirming that geometric sensitivity primarily tracks high-frequency edge content rather than true hallucination.
2. **Multi-Temporal Sensitivity is the Primary Signal:** Measuring variance across diverse Sentinel-2 temporal revisits ($U_{\text{input}}$) provides a powerful, robust error-ranking signal ($\bar{\rho} = 0.2754$, $+37.3\%$ gain over baseline), accounting for over $90\%$ of the paired advantage over the spatial edge detector.
3. **The Hybrid Estimator Generalizes Reliably:** Blending normalized TTA and input sensitivity with frozen $\alpha^* = 0.1900$ achieves superior error-ranking correlation on unseen holdout scenes ($\bar{\rho} = 0.2800$, $+39.53\%$ relative gain over Sobel, $p < 10^{-13}$, Cohen's $d = 0.83$, $86.92\%$ win rate), replicating the development effect size within $0.003$ correlation units.
4. **Nuanced Role of TTA:** While the Hybrid estimator achieves higher median correlation and beats Input Sensitivity on $66.4\%$ of holdout scenes ($p = 0.00059$, Wilcoxon), parametric significance ($t$-test $p = 0.1747$) is sensitive to isolated negative outliers. TTA should be viewed as a modest regularizer rather than an independent primary estimator.

All supporting code, frozen parameters, raw per-tile metrics, recomputed audit tables, and publication-ready vector figures are completely archived and reproducible.
