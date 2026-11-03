# EXP-019: Formal Erratum for EXP-017 Development Reference Statistics

**Date:** 2026-10-23  
**Target Document:** [`results/satlas_sr_experiment_017_locked_holdout/EXP017_LOCKED_HOLDOUT_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/EXP017_LOCKED_HOLDOUT_REPORT.md)  
**Associated Data File:** [`results/satlas_sr_experiment_017_locked_holdout/summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json)  
**Status of Prior Artifacts:** The original EXP-017 report and summary files remain **strictly preserved and unchanged** to maintain forensic provenance. This erratum document serves as the formal official record of correction.

---

## 1. Description of Error

In **EXP-017 (Frozen Independent Holdout Evaluation of Hybrid Uncertainty)**, Section 5 ("Development vs Holdout Replication Analysis") and `summary.json` (lines 312–319 under `"development_exp015_cv"`) contained inaccurate numerical values for the historical **Development Partition ($N=107$ tiles)** comparison baseline.

Specifically:
1. The development per-tile win rate of Hybrid Uncertainty over the Sobel Gradient Baseline was reported as **$77.57\%$ ($83/107$ tiles)**. The verified ground truth from EXP-015 and EXP-016 is **$89.72\%$ ($96/107$ tiles)**.
2. The development Hybrid mean Spearman correlation was reported as **$0.3048$** (or $0.30476$). The verified ground truth is **$0.304210$**.
3. The development Sobel mean Spearman correlation was reported as **$0.2224$** (or $0.22241$). The verified ground truth is **$0.222005$**.
4. The paired mean difference was reported as **$+0.0824$** (or $+0.08235$). The verified ground truth is **$+0.082205$**.

---

## 2. Root Cause Analysis

During the execution of EXP-017, the evaluation runner script [`run_holdout_evaluation.py:L577-L584`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/run_holdout_evaluation.py#L577-L584) generated the reproduction summary dictionary. Instead of reading the development statistics dynamically from [`results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json) or [`per_tile_oof_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/per_tile_oof_metrics.csv), the script hardcoded the development block:

```python
# run_holdout_evaluation.py (Lines 577-584):
"development_vs_holdout_replication": {
    "development_exp015_cv": {
        "mean_rho_hybrid": 0.30476,       # Discrepancy: +0.00055 vs true 0.304210
        "mean_rho_sobel": 0.22241,        # Discrepancy: +0.00040 vs true 0.222005
        "paired_mean_delta": 0.08235,     # Discrepancy: +0.00014 vs true 0.082205
        "relative_gain_pct": 37.03,
        "t_p_value": 3.20e-11,
        "win_rate_pct": 77.57             # Discrepancy: -12.15% vs true 89.72% (96/107)
    },
    ...
}
```

The $77.57\%$ ($83/107$) figure originated from an early exploratory prototype comparison on raw unstandardized features. When drafting the script, this approximate value was manually transcribed into the summary block.

---

## 3. Verified Numerical Replacements

The table below reconciles the reported values against the audited ground-truth data from EXP-015 and EXP-016 (re-verified in EXP-018):

| Metric Field | As Reported in EXP-017 | Verified Ground Truth | Source Artifact for Ground Truth |
| :--- | :---: | :---: | :--- |
| **Hybrid Mean Spearman $\rho$** | $0.30476$ | **$0.304210$** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L101` |
| **Sobel Baseline Mean $\rho$** | $0.22241$ | **$0.222005$** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L88` |
| **Paired Mean Difference ($\bar{\Delta}$)** | $+0.08235$ | **$+0.082205$** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L226` |
| **Paired Student's $t$-statistic** | $7.4107$ | **$7.4107$** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L227` |
| **Paired Student's $t$ $p$-value** | $3.20 \times 10^{-11}$ | **$3.20 \times 10^{-11}$** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L228` |
| **Paired Wilcoxon Signed-Rank $W$** | Not reported in table | **$336.0$ ($p = 2.11 \times 10^{-15}$)** | `results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L229-L230` |
| **Per-Tile Wins / Losses / Ties** | $83$ wins / $24$ losses | **$96$ wins / $11$ losses / $0$ ties** | `results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv:L2` |
| **Per-Tile Win Rate (%)** | $77.57\%$ | **$89.72\%$** | `results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv:L2` |
| **Cohen's $d$ Effect Size** | $0.7428$ | **$0.7164$** | `results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv:L2` |

---

## 4. Scope and Scientific Impact Assessment

1. **Holdout Evaluation Integrity Unaffected:**
   - The holdout evaluation pipeline did not read or depend upon the transcribed development summary dictionary. All holdout predictions, uncertainty maps, registration shifts, and holdout statistics were computed directly from holdout raw imagery and are 100% verified and unaffected by this erratum.
2. **Replication Interpretation Strengthened:**
   - Correcting the development win rate from $77.57\%$ to the true $89.72\%$ reveals that the development and holdout partitions are in even tighter agreement than initially reported:
     - Development Win Rate: **$89.72\%$ (96/107)**
     - Holdout Win Rate: **$86.92\%$ (93/107)**
   - The paired treatment effect is virtually identical across partitions ($+0.0822$ on development vs $+0.0793$ on holdout), differing by just $0.0029$ correlation units.
3. **Action Taken:**
   - The verified ground-truth values have been integrated into [`EXP019_RESULTS_TABLES.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_RESULTS_TABLES.csv) and [`EXP019_RESEARCH_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_RESEARCH_REPORT.md).
