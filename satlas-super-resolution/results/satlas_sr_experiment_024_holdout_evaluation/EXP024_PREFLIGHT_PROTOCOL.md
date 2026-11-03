# EXP-024: Preflight Protocol Freeze Document

**Experiment ID:** EXP-024  
**Date:** October 27, 2026  
**Repository:** `satlas-super-resolution`  
**Target Partition:** Locked Independent Holdout Set ($N = 107$ tiles)  
**Status:** FROZEN — PENDING SINGLE EVALUATION PASS  

---

## 1. Scientific Objective and Context

Following the success of development-partition evaluations:
- **EXP-020A:** Demonstration of uncertainty-guided reconstruction improvements via discrete top-$K$ replacement on 107 development tiles.
- **EXP-021:** Independent result integrity audit verifying development findings with zero holdout leakage.
- **EXP-022:** Development robustness analysis across spatial folds, budgets, and observable scene features.
- **EXP-023:** Introduction of continuous soft-blending, demonstrating elimination of artificial boundary seams and superior error-sharpness tradeoff over hard replacement.

The objective of **EXP-024** is to conduct a **strictly preregistered, frozen holdout evaluation** of continuous uncertainty-guided soft blending on the 107 unseen holdout tiles.

---

## 2. Mandatory Protocol Safeguards

1. **Strict Partition Isolation:**
   - Holdout tiles: Exactly $N = 107$ tiles specified in `split_manifest.json` and `EXP-017`.
   - Zero overlap with development tiles ($N = 107$) or quarantined tiles ($N = 42$).
   - Quarantined tiles are permanently excluded.
2. **Frozen Hyperparameters (No Holdout Tuning):**
   - No hyperparameter tuning, model training, or threshold adjustments against holdout data.
   - All parameters ($\mu_{\text{tta}}, \sigma_{\text{tta}}, \mu_{\text{input}}, \sigma_{\text{input}}, \alpha, \beta, \text{budget}$) are identical to those frozen on the development set in EXP-023.
3. **Single Evaluation Pass:**
   - The holdout evaluation script will execute exactly once, reading inputs deterministically and producing final outputs.
4. **Historical Artifact Immutability:**
   - Previous experiment folders (EXP-001 through EXP-023) remain completely untouched and unmodified.

---

## 3. Preregistered Formulations and Parameters

### 3.1 Continuous Soft Blending Operator
For interior crop pixels $(x, y) \in [4, 124) \times [4, 124)$:
$$P_{\text{soft}}(x, y) = (1 - w(x, y)) P_0(x, y) + w(x, y) P_e(x, y)$$
where:
- $P_0(x, y)$: Baseline $S_0$ prediction (single clear frame).
- $P_e(x, y) = \frac{1}{6} \sum_{k \in \{0, 1, 3, 4, 5, 6\}} S_k(x, y)$: Six-prediction temporal ensemble mean.
- $w(x, y) = \sigma\left(\frac{U_{\text{hybrid}}(x, y) - \tau^*}{\beta}\right)$: Continuous blending weight.

### 3.2 Frozen Hybrid Uncertainty Estimator
$$U_{\text{hybrid}}(x, y) = \alpha \cdot Z(U_{\text{tta}}(x, y)) + (1 - \alpha) \cdot Z(U_{\text{input}}(x, y))$$
with frozen development parameters:
- $\alpha = 0.1900$, $1 - \alpha = 0.8100$
- $Z(U_{\text{tta}}) = \frac{U_{\text{tta}} - 0.083126}{0.051388}$
- $Z(U_{\text{input}}) = \frac{U_{\text{input}} - 0.057286}{0.041238}$

### 3.3 Root-Finding Budget Matching
The threshold $\tau^*$ is uniquely and deterministically solved per tile such that the mean blending weight matches the target budget $B$:
$$F(\tau^*) = \frac{1}{14400} \sum_{x, y} \sigma\left(\frac{U_{\text{hybrid}}(x, y) - \tau^*}{\beta}\right) - B = 0$$
- Primary Budget: $B = 0.20$ (2,880 effective replaced pixels).
- Secondary Budgets: $B \in \{0.10, 0.30\}$.
- Primary Temperature: $\beta = 0.20$.
- Sensitivity Temperatures: $\beta \in \{0.05, 0.10, 0.20, 0.40\}$.
- Solver: Deterministic 1D Brent root finding bracketed on $[\min(U) - 15\beta, \max(U) + 15\beta]$ with numerical tolerance $10^{-6}$.

### 3.4 Spatial Alignment Policy
- Registration integer shifts $(\Delta x, \Delta y) \in [-8, 8]^2$ are loaded from the precomputed EXP-017 holdout NPZ files (`results/satlas_sr_experiment_017_locked_holdout/maps/{tile}_holdout_maps.npz`).
- Evaluation window: $120 \times 120$ central crop ($[4, 124) \times [4, 124)$), matching the cPSNR standard.

---

## 4. Prespecified Endpoints and Hypotheses

### Primary Endpoint
**Continuous Soft Blending (Budget 20%, $\beta = 0.20$) vs. Discrete Hard Replacement (Top 20%):**
- **Null Hypothesis ($H_0$):** $\mu_{\Delta \text{MAE}} = 0$ on the 107 holdout tiles.
- **Alternative Hypothesis ($H_1$):** $\mu_{\Delta \text{MAE}} < 0$ (Continuous soft blending yields lower MAE than hard replacement).
- Significance threshold: $\alpha = 0.05$ (two-tailed paired Student's t-test and Wilcoxon signed-rank test).

### Secondary Endpoints
1. **Soft 20% vs. Baseline $S_0$:** Verification of statistically significant reduction in MAE and increase in PSNR.
2. **Hard 20% vs. Baseline $S_0$:** Holdout confirmation of the discrete top-$K$ improvement reported in EXP-020A.
3. **Soft 20% vs. Full Temporal Ensemble $P_e$:** Assessment of high-frequency sharpness retention (Sobel gradient magnitude) vs. pixel error.
4. **Boundary Seam Gradient Discontinuity:** Verification of seam artifact suppression on the 3-pixel morphological boundary band.
5. **Weight Map Smoothness:** Verification of spatial gradient magnitude reduction $\|\nabla w_{\text{hard}}\| - \|\nabla w_{\text{soft}}\|$.
6. **Sensitivity Analyses:** Holdout stability across budgets ($10\%, 20\%, 30\%$) and temperature values $\beta \in \{0.05, 0.10, 0.20, 0.40\}$.

---

## 5. Deliverables Specification

1. `preflight_protocol.json` — Machine-readable parameter freeze.
2. `EXP024_PREFLIGHT_PROTOCOL.md` — This freeze document.
3. `run_holdout_blending_evaluation.py` — Deterministic evaluation script.
4. `per_tile_holdout_metrics.csv` — Full metrics per tile ($N = 107$).
5. `aggregate_statistics.csv` — Formal paired comparisons, t-tests, Wilcoxon tests, effect sizes, win rates.
6. `boundary_gradient_diagnostics.csv` — Seam gradient analysis on holdout.
7. `beta_sensitivity_holdout.csv` — Temperature grid evaluation.
8. `audit_checks.json` — Quality-control validation checklist (7/7 gates).
9. `source_manifest.csv` — SHA-256 integrity manifest.
10. `run_command_record.txt` — Command execution provenance.
11. `fig1_holdout_mae_psnr.svg` — Vector figure of MAE and PSNR tradeoffs.
12. `fig2_holdout_boundary_discontinuity.svg` — Vector figure of seam discontinuity suppression.
13. `EXP024_HOLDOUT_REPORT.md` — Comprehensive research report.
