# EXP-019: Publication Figure Plan

**Project:** Satlas-SR Uncertainty-Estimation Evaluation  
**Deliverable:** Publication-Ready Figure Specifications and Generated Artifacts  
**Target Directory:** `results/satlas_sr_experiment_019_research_report/`  
**Date:** 2026-10-23  

---

## 1. Overview of Publication Figures

Four standalone publication-grade figures have been planned, specified, and generated directly into this experiment folder without running additional model inference. Each figure addresses a core question in the research report, adheres to strict color-contrast standards, and cites exact underlying disk artifacts.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PUBLICATION FIGURE INVENTORY                            │
├──────────┬───────────────────────────────────────────────┬─────────────────────────────┤
│ Figure   │ Title                                         │ Formats Generated           │
├──────────┼───────────────────────────────────────────────┼─────────────────────────────┤
│ Figure 1 │ Correlation Distributions on Locked Holdout   │ `fig1_correlation_dist.svg` │
│ Figure 2 │ Paired Treatment Effect (Dev vs Holdout)      │ `fig2_paired_diff.svg`      │
│ Figure 3 │ Sparsification Error-Retention Curves         │ `fig3_sparsification.svg`   │
│ Figure 4 │ Discrepancy Analysis (Hybrid vs Input Sens.)  │ `fig4_hybrid_vs_input.svg`  │
└──────────┴───────────────────────────────────────────────┴─────────────────────────────┘
```

---

## 2. Detailed Figure Specifications

### Figure 1: Per-Tile Spearman Error-Ranking Correlation Distributions (Holdout, $N=107$)
- **File Artifacts:**
  - Vector SVG: [`fig1_correlation_distributions.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig1_correlation_distributions.svg)
  - Bitmap Preview: [`fig1_correlation_distributions.png`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig1_correlation_distributions.png)
- **Scientific Objective:** Compare the marginal error-ranking performance across all four estimators on the unseen holdout partition.
- **Source Data:**
  - Artifact: `results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv`
  - Columns: `rho_hybrid_comp`, `rho_input_comp`, `rho_grad_comp`, `rho_tta_comp`
- **Visualization Type:** Box-and-whisker plot with parametric mean overlay markers (red circles) and median dividers (solid bars).
- **Key Statistics Displayed:**
  - **Hybrid Estimator:** Mean = $0.2800$, Median = $0.2957$, IQR = $[0.201, 0.380]$
  - **Input Sensitivity:** Mean = $0.2754$, Median = $0.2901$, IQR = $[0.189, 0.370]$
  - **Sobel Gradient:** Mean = $0.2006$, Median = $0.2115$, IQR = $[0.100, 0.300]$
  - **$D_4$ TTA:** Mean = $0.1700$, Median = $0.1713$, IQR = $[0.080, 0.270]$
- **Caption for Manuscript:**
  > *Figure 1: Distribution of per-tile Spearman rank correlations ($\rho$) between predicted pixel uncertainty and compensated reconstruction absolute errors on the locked 107-tile holdout partition. Horizontal bold lines denote sample medians; red circular markers denote parametric means; outer whiskers denote minimum and maximum observations.*

---

### Figure 2: Paired Treatment Effect (Hybrid vs Sobel Baseline) Across Partitions
- **File Artifacts:**
  - Vector SVG: [`fig2_paired_differences.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig2_paired_differences.svg)
  - Bitmap Preview: [`fig2_paired_differences.png`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig2_paired_differences.png)
- **Scientific Objective:** Demonstrate the stability of the primary paired treatment effect ($\Delta \rho = \rho_{\text{hybrid}} - \rho_{\text{grad}}$) and show replication fidelity between development and holdout.
- **Source Data:**
  - Development Data: `results/satlas_sr_experiment_015_hybrid_uncertainty/per_tile_oof_metrics.csv`
  - Holdout Data: `results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv`
  - Reconciled Audit: `results/satlas_sr_experiment_018_holdout_consistency_audit/reconciled_development_statistics.csv`
- **Visualization Type:** Point-and-whisker paired effect size chart with 95% Student's $t$ confidence intervals and per-tile win counts.
- **Key Statistics Displayed:**
  - **Development (5-Fold CV, $N=107$):** $\bar{\Delta} = +0.082205$, 95% CI $[+0.0602, +0.1042]$, Win Rate: **$89.72\%$ (96/107)**
  - **Holdout (Frozen Single Pass, $N=107$):** $\bar{\Delta} = +0.079316$, 95% CI $[+0.0611, +0.0975]$, Win Rate: **$86.92\%$ (93/107)**
- **Caption for Manuscript:**
  > *Figure 2: Replicated paired treatment effect of the Hybrid Uncertainty estimator over the Sobel gradient baseline. Points denote paired sample mean differences ($\bar{\Delta}$); error bars denote 95% parametric confidence intervals. Win rates indicate the proportion of individual geographic scenes where Hybrid uncertainty yielded higher error-ranking correlation than the spatial gradient baseline.*

---

### Figure 3: Sparsification Error-Retention Curves (Holdout Partition)
- **File Artifacts:**
  - Vector SVG: [`fig3_sparsification_curves.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig3_sparsification_curves.svg)
  - Bitmap Preview: [`fig3_sparsification_curves.png`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig3_sparsification_curves.png)
- **Scientific Objective:** Illustrate the practical risk-reduction utility of discarding high-uncertainty pixels in downstream workflows.
- **Source Data:**
  - Primary Summary: `results/satlas_sr_experiment_017_locked_holdout/summary.json`
  - Integration Definition: Numerical trapezoidal integration over 20 uniform fractions $p \in [0.0, 0.95]$
- **Visualization Type:** Monotonic decay curve of remaining Mean Absolute Error (MAE) as a function of pixel removal fraction $p$.
- **Key Statistics Displayed:**
  - **Random Sparsification (Flat):** Constant baseline at $\text{MAE} = 0.0573$
  - **Sobel Gradient:** MAE reduced by $8.43\%$ at $p=0.20$; $\text{AURG} = 0.01124$
  - **Hybrid Estimator:** MAE reduced by $11.27\%$ at $p=0.20$; $\text{AURG} = 0.01593$
  - **Shaded Area:** Represents AURG gain ($+0.00469$, $p < 10^{-13}$)
- **Caption for Manuscript:**
  > *Figure 3: Practical error sparsification on the holdout partition. Curves depict the average reconstruction Mean Absolute Error (MAE) of remaining pixels after removing fraction $p$ of the highest-uncertainty pixels. Shaded blue region represents the gain in Area Under Random Error Retention (AURG) achieved by the Hybrid estimator over random pixel rejection.*

---

### Figure 4: Distribution of Differences (Hybrid vs Input Sensitivity) and Statistical Caution
- **File Artifacts:**
  - Vector SVG: [`fig4_hybrid_vs_input_distribution.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig4_hybrid_vs_input_distribution.svg)
  - Bitmap Preview: [`fig4_hybrid_vs_input_distribution.png`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/fig4_hybrid_vs_input_distribution.png)
- **Scientific Objective:** Transparently diagnose the divergence between parametric ($p = 0.1747$) and non-parametric ($p = 0.00059$) inferential tests for Hybrid vs Input Sensitivity.
- **Source Data:**
  - Holdout Per-Tile Data: `results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv`
  - Audit Diagnosis: `results/satlas_sr_experiment_018_holdout_consistency_audit/EXP018_CONSISTENCY_AUDIT_REPORT.md`
- **Visualization Type:** Difference density diagram highlighting positive/negative partitions, distribution median, mean, and the influential negative outlier.
- **Key Statistics Displayed:**
  - **Positive Wins:** $71 / 107$ tiles ($66.36\%$, nearly $2:1$ ratio)
  - **Sample Median:** $+0.007191$
  - **Sample Mean:** $+0.004538$
  - **Skewness:** $-4.6405$, **Kurtosis:** $36.1496$
  - **Single Negative Outlier:** $-0.269097$ (Tile `32822_54029`)
- **Caption for Manuscript:**
  > *Figure 4: Empirical distribution of paired differences ($\Delta = \rho_{\text{hybrid}} - \rho_{\text{input}}$) across the 107 holdout scenes. In 71 tiles (66.4%), the Hybrid estimator outperforms Input Sensitivity alone, driving a highly significant Wilcoxon signed-rank test ($p = 0.00059$). However, an isolated severe negative outlier (tile 32822_54029, $\Delta = -0.2691$) inflates sample variance by 68%, reducing the paired t-test statistic to 1.37 ($p = 0.1747$). Inference therefore depends on test assumptions.*
