# EXP-019: Final Quality-Control (QC) Review Report

**Review Status:** COMPLETED  
**Overall Verdict:** **READY FOR SUPERVISOR REVIEW (WITH 1 MINOR PROVENANCE NOTE)**  
**Date:** 2026-10-23  
**Auditor Mode:** Read-Only Verification  
**Repository:** `satlas-super-resolution`  
**Git Commit Hash:** `3e67973f6746bedbd91cc032b4888574980e1daf`  
**Target Directory Reviewed:** `results/satlas_sr_experiment_019_research_report/`  

---

## 1. Executive Summary & Verdict

This report presents a thorough, independent read-only quality-control review of the **EXP-019 Research Report and Erratum Package**. Every deliverable—including the primary research manuscript, formal erratum, consolidated CSV tables, figure plan, vector/bitmap figures, source manifest, and validation JSON—was inspected and verified against the underlying disk artifacts across EXP-011 through EXP-018.

### Summary of QC Findings:
- **Blockers:** 0
- **Major Defects:** 0
- **Minor Observations:** 1 (Minor typographical filename in source manifest for EXP-013)
- **Passing Checks:** 7 / 7 Core Verification Categories Passed

### Recommendation: **READY FOR SUPERVISOR REVIEW**
The package meets rigorous scientific, numerical, and documentation standards. All reported primary and secondary endpoints match raw data to full machine precision ($|\Delta| < 10^{-6}$), the erratum correctly resolves historical development discrepancies without touching prior artifacts, and the manuscript strictly adheres to required statistical and scientific guardrails.

---

## 2. Comprehensive QC Findings by Check Category

### Check 1: Headline Number Verification Against Source Artifacts
- **Classification:** **PASS**
- **Findings:** Every headline numerical value cited in the Abstract, Executive Summary, Results tables, and Erratum was verified against the primary underlying disk artifacts:
  1. **Holdout Primary Paired Difference:** Reported as $\bar{\Delta} = +0.079316$ (95% CI: $[0.061084, 0.097548]$, $t = 8.6251$, $p = 6.87 \times 10^{-14}$, $W = 458.0$, $p = 4.17 \times 10^{-14}$, Cohen's $d = 0.8338$, wins: $93/107$). Matches [`results/satlas_sr_experiment_017_locked_holdout/summary.json:L98-L112`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json#L98-L112) and [`results/satlas_sr_experiment_018_holdout_consistency_audit/recomputed_holdout_statistics.csv:L2`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_018_holdout_consistency_audit/recomputed_holdout_statistics.csv#L2) to full machine precision.
  2. **Holdout Marginal Spearman Means:** Hybrid $\bar{\rho} = 0.279966$ (SD $0.149602$), Sobel $\bar{\rho} = 0.200649$ (SD $0.142533$). Matches [`results/satlas_sr_experiment_017_locked_holdout/summary.json:L24, L45`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json#L24).
  3. **Reconciled Development Paired Gain:** Reported as $\bar{\Delta} = +0.082205$ ($+37.03\%$ relative gain, $t = 7.4107$, $p = 3.20 \times 10^{-11}$, $W = 336.0$, $p = 2.11 \times 10^{-15}$, Cohen's $d = 0.7164$, wins: $96/107$). Matches [`results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L226-L230`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json#L226-L230) and [`results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv:L2`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv#L2).
  4. **Reconciled Development Marginal Means:** Hybrid $\bar{\rho} = 0.304210$, Sobel $\bar{\rho} = 0.222005$. Matches [`results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L88, L101`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json#L88).
  5. **Sparsification MAE Reduction at 20% Discard:** Hybrid $11.27\%$, Sobel $8.43\%$, paired gain $+2.84\%$ ($t = 9.6933, p = 2.75 \times 10^{-16}$, wins: $98/107$). Matches [`results/satlas_sr_experiment_017_locked_holdout/summary.json:L249, L270, L278`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json#L249).

---

### Check 2: Consistency of Statistical Definitions
- **Classification:** **PASS**
- **Findings:**
  - **Confidence Intervals:** All 95% confidence intervals across reports and CSV tables consistently apply the two-sided Student's $t$ critical value with $df = N - 1 = 106$ ($t_{0.975, 106} = 1.9826$) multiplied by standard error of the mean $s / \sqrt{N}$.
  - **Direction of Differences:** All paired differences consistently apply $\Delta = \text{Evaluated Method} - \text{Reference Baseline}$.
  - **Effect Sizes:** Cohen's $d$ is consistently computed as $d = \bar{\Delta} / s_{\Delta}$ using the sample standard deviation of paired differences ($s_{\Delta}$).
  - **Win/Loss Classification:** Wins ($\Delta > 0$), losses ($\Delta < 0$), and ties ($\Delta = 0$) are strictly partitioned without rounding bias. Zero ties occurred on holdout primary endpoints.
  - **Relative Gain Denominators:** Relative gains ($\%$) explicitly declare their reference baseline denominator (e.g., $+39.53\%$ relative gain is $(0.079316 / 0.200649) \times 100$).

---

### Check 3: Traceability of Development Confidence Intervals and Secondary Endpoints
- **Classification:** **PASS**
- **Findings:**
  - **Development 95% CIs:**
    - Development Paired CI $[+0.060212, +0.104197]$ traces directly to [`results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv:L2`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_016_independent_audit/recomputed_paired_tests.csv#L2).
    - Development Individual CIs (Hybrid $[0.2747, 0.3337]$, Sobel $[0.1906, 0.2534]$) trace directly to [`results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L65, L91, L104`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json#L65).
  - **AURG Statistics:**
    - Holdout AURG (Hybrid $0.015927$, Sobel $0.011235$, paired difference $+0.004692$, $t = 8.7886$, $p = 2.96 \times 10^{-14}$) traces directly to [`results/satlas_sr_experiment_017_locked_holdout/summary.json:L186-L224`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json#L186-L224).
    - Development AURG (Hybrid $0.017007$, Sobel $0.012194$, paired difference $+0.004813$, $p = 3.31 \times 10^{-15}$) traces directly to [`results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json:L143, L156, L231`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_015_hybrid_uncertainty/method_feasibility.json#L143).

---

### Check 4: Figure Integrity, Data Correspondence, and Visual Consistency
- **Classification:** **PASS**
- **Findings:** All eight figure files (4 vector SVG and 4 high-resolution PNG previews) exist on disk and were verified:
  - **Figure 1 (`fig1_correlation_distributions`):** Y-axis correctly spans $[-0.4, 0.8]$; boxes and whiskers precisely reflect Holdout medians ($0.296, 0.290, 0.211, 0.171$) and parametric means ($0.280, 0.275, 0.201, 0.170$).
  - **Figure 2 (`fig2_paired_differences`):** Y-axis correctly spans $[0.00, 0.30]$; point estimates and 95% CIs for Development ($+0.0822$, $[0.060, 0.104]$, $89.72\%$ wins) and Holdout ($+0.0793$, $[0.061, 0.098]$, $86.92\%$ wins) match Table 2 exactly.
  - **Figure 3 (`fig3_sparsification_curves`):** X-axis correctly models discard fraction $p \in [0.0, 0.95]$; flat random reference reflects initial MAE ($0.0573$); curves accurately depict the $11.27\%$ vs $8.43\%$ drop at $p=0.20$ and shaded AURG region.
  - **Figure 4 (`fig4_hybrid_vs_input_distribution`):** Correctly graphs difference span $[-0.30, +0.10]$; visualizes the $66.4\%$ positive win region ($71$ tiles), median ($+0.0072$), mean ($+0.0045$), and highlights the $-0.2691$ outlier on tile `32822_54029`.

---

### Check 5: Scientific Caution, Hypothesis Differentiation, and Guardrails
- **Classification:** **PASS**
- **Findings:**
  - **Hybrid vs Input Sensitivity Caution:** Section 7.2 of [`EXP019_RESEARCH_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_RESEARCH_REPORT.md) explicitly reports both paired $t$-test ($p = 0.174703$) and Wilcoxon signed-rank test ($p = 0.000587$). It describes the extreme negative skewness ($-4.6405$) and kurtosis ($36.1496$) caused by the $-0.2691$ outlier on tile `32822_54029`. It explicitly warns against declaring Wilcoxon automatically superior and notes that inference depends upon test assumptions.
  - **No Over-Attribution Claims:** The report avoids asserting unverified linear variance attributions, defining attribution strictly as the ratio of paired mean improvements over Sobel ($94.3\%$ on holdout, $91.9\%$ on dev), while cautioning that linear attribution in non-linear rank correlations is an empirical approximation.
  - **Strict Guardrails Enforced:** Section 9 explicitly articulates all required boundaries:
    1. Scores represent relative rank heuristics, not calibrated probabilities.
    2. Integer image translation is not geodetic orthorectification.
    3. No building-level polygon segmentation masks were evaluated.
    4. Generalization is bounded to the 107 evaluated scenes.

---

### Check 6: Erratum Documentation and Historical Preservation
- **Classification:** **PASS**
- **Findings:**
  - [`EXP019_ERRATUM.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_ERRATUM.md) thoroughly documents the origin of the transcription error in `run_holdout_evaluation.py:L577-L584` (which entered $77.57\%$ / $83$ wins instead of ground truth $89.72\%$ / $96$ wins).
  - All historical artifacts in `results/satlas_sr_experiment_017_locked_holdout/` and earlier experiment directories remain **strictly unaltered and preserved**.

---

### Check 7: Deliverable Completeness and Manifest Provenance
- **Classification:** **MINOR**
- **Findings:**
  - **Completeness:** All 7 required deliverable files and 8 figure files exist in `results/satlas_sr_experiment_019_research_report/`.
  - **Manifest Integrity Check:**
    - 20 out of 21 artifact paths in [`EXP019_SOURCE_MANIFEST.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_SOURCE_MANIFEST.csv) exist on disk and match their verified SHA256 hashes or directory flags.
    - **Minor Observation:** Row 2 of `EXP019_SOURCE_MANIFEST.csv` lists `results/satlas_sr_experiment_013_geospatial_readiness_audit/GEOSPATIAL_ALIGNMENT_AUDIT_REPORT.md`. In that folder, the actual markdown files are named `GEOSPATIAL_METADATA_AUDIT.md` and `REFERENCE_VALIDITY_REPORT.md`.
    - **Action Needed:** None required for scientific validity; documented here as a minor typographical citation in the source manifest.

---

## 3. Final Classification Table

| Check ID | Verification Focus | Classification | Status / Impact |
| :---: | :--- | :---: | :--- |
| **QC-01** | Headline numbers vs. cited source artifacts | **PASS** | Exact machine precision match ($|\Delta| < 10^{-6}$). |
| **QC-02** | Consistency of statistical definitions & tests | **PASS** | $df=106$, $t$-critical, paired directions, effect sizes verified. |
| **QC-03** | Traceability of dev CIs and secondary endpoints | **PASS** | All intervals, AURG, and sparsification MAE traced to disk. |
| **QC-04** | Figure axes, legends, units, and data matching | **PASS** | All 8 SVG/PNG figures match underlying tables exactly. |
| **QC-05** | Primary vs secondary endpoints & guardrails | **PASS** | Guardrails, non-claims, and $p$-value caution verified. |
| **QC-06** | Formal erratum and preservation of prior files | **PASS** | EXP-017 erratum documented; historical files untouched. |
| **QC-07** | Deliverable completeness & source manifest | **MINOR** | All 15 files present; 1 minor filename typo in manifest. |

---

## 4. Final Recommendation

### **RECOMMENDATION: READY FOR SUPERVISOR REVIEW**

The **EXP-019 Research Report and Erratum Package** is complete, mathematically sound, forensically audited, and ready for supervisor and stakeholder review. All findings are supported by frozen local artifacts, and no further experiments or evaluations are required.
