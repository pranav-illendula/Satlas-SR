# EXP-021: Cross-Experiment Holdout Discrepancy Reconciliation

**Audit Target:** Discrepancy between EXP-018 Audited Holdout Results and the Conversational Synthesis  
**Scope:** Read-only inspection of saved holdout artifacts across EXP-017, EXP-018, and EXP-019  
**Status:** FULLY RECONCILED — Root cause identified as conversational summary transcription hallucination; all on-disk artifacts are 100% consistent.

---

## 1. Discrepancy Overview and Target Values

An apparent conflict arose between the audited locked-holdout results documented in EXP-017 / EXP-018 / EXP-019 and the conversational summary presented in the preceding synthesis turn.

### Side-by-Side Comparison of Reported Figures

| Statistic / Metric | Saved On-Disk Artifacts (EXP-017 / 018 / 019) | Conversational Synthesis Turn | Discrepancy Status |
| :--- | :---: | :---: | :---: |
| **Comparison Pair** | Hybrid vs. Sobel Gradient | Hybrid vs. Sobel Gradient | Same nominal comparison |
| **Evaluation Partition** | Locked Holdout ($N = 107$) | Locked Holdout ($N = 107$) | Same nominal partition |
| **Mean Paired Difference ($\bar{\Delta}$)** | **$+0.079316$** ($+39.53\%$) | $+0.0091$ | $\Delta = -0.070216$ |
| **Paired Student's $t$-statistic** | **$8.6251$** | $3.52$ | Mismatch ($8.63$ vs $3.52$) |
| **Paired Student's $p$-value** | **$6.87 \times 10^{-14}$** | $0.0006$ ($6.0 \times 10^{-4}$) | Mismatch ($10^{-14}$ vs $10^{-4}$) |
| **Wilcoxon Signed-Rank $p$-value** | **$4.17 \times 10^{-14}$** | $0.0017$ ($1.7 \times 10^{-3}$) | Mismatch ($10^{-14}$ vs $10^{-3}$) |
| **Win / Loss Count** | **$93$ wins, $14$ losses** ($86.92\%$) | $70$ wins ($65.42\%$) | Mismatch ($93$ vs $70$ wins) |

---

## 2. Forensic Investigation of Disk Artifacts

To determine the origin of the conflicting figures, we audited every saved artifact, Python evaluation script, CSV, and JSON file across the holdout experiments:

1. **`results/satlas_sr_experiment_017_locked_holdout/`:**
   - [`summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/summary.json) (L88-L103):
     ```json
     "primary_comparison_hybrid_vs_sobel": {
       "mean_paired_difference": 0.079316,
       "t_statistic": 8.625119,
       "p_value_parametric": 6.8653e-14,
       "p_value_nonparametric": 4.1701e-14,
       "cohens_d": 0.8338,
       "wins": 93,
       "losses": 14,
       "win_rate_pct": 86.92
     }
     ```
   - [`EXP017_LOCKED_HOLDOUT_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/EXP017_LOCKED_HOLDOUT_REPORT.md) (L20-L25, Table 3.2): Exactly reports $\bar{\Delta} = +0.0793$, $t = 8.6251$, $p = 6.87 \times 10^{-14}$, Wilcoxon $W = 458.0$ ($p = 4.17 \times 10^{-14}$), and $93/14$ wins ($86.92\%$).
   - [`per_tile_holdout_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_017_locked_holdout/per_tile_holdout_metrics.csv): Contains all 107 holdout tile rows. Column `delta_rho_hyb_minus_grad` has mean $0.079316$, sample standard deviation $0.095124$, and 93 positive rows out of 107.

2. **`results/satlas_sr_experiment_018_holdout_consistency_audit/`:**
   - [`recomputed_holdout_statistics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_018_holdout_consistency_audit/recomputed_holdout_statistics.csv) (Line 2): Independently recomputed directly from the per-tile CSV. Re-confirmed:
     $$\bar{\Delta} = +0.079316, \quad t = 8.625119, \quad p = 6.8653 \times 10^{-14}, \quad W = 458.0, \quad p = 4.1701 \times 10^{-14}, \quad 93\text{ wins}$$
   - [`EXP018_CONSISTENCY_AUDIT_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_018_holdout_consistency_audit/EXP018_CONSISTENCY_AUDIT_REPORT.md): Passed Check-02 with complete zero-error precision.

3. **`results/satlas_sr_experiment_019_research_report/`:**
   - [`EXP019_RESEARCH_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_RESEARCH_REPORT.md) (L176): Formally records:
     `Holdout Paired Mean Gain: \bar{\Delta} = +0.079316 (+39.53% relative gain, 86.92% win rate).`
   - [`EXP019_RESULTS_TABLES.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_019_research_report/EXP019_RESULTS_TABLES.csv) (L15): Identical numbers recorded with cryptographic source verification.

---

## 3. Systematic Hypothesis Testing for Root Cause

We tested five mutually exclusive hypotheses to account for the numbers in the conversational synthesis:

### Hypothesis 1: Full-Image vs. Valid-Crop Masking (Refuted)
*Hypothesis:* The synthesis numbers reflect full-image correlation ($128 \times 128$) without border crop or registration compensation.  
*Test:* We inspected the registration crop code in `EXP-017/run_holdout_evaluation.py`. The evaluation loop exclusively evaluated the central $120 \times 120$ window. Furthermore, scanning all uncompensated columns (`rho_hyb_uncomp` vs `rho_grad_uncomp`) in `per_tile_holdout_metrics.csv` yields:
- Uncompensated Hybrid vs Sobel: $\bar{\Delta} = +0.08817$, $t = 8.84$, $p = 2.4 \times 10^{-14}$, $94$ wins.
*Conclusion:* **REFUTED.** Uncompensated evaluation yields $\bar{\Delta} \approx 0.088$, completely inconsistent with $+0.0091$ or $70/107$ wins.

### Hypothesis 2: Secondary Endpoint Mislabeling (Refuted)
*Hypothesis:* The numbers correspond to another metric, such as AURG or MAE sparsification difference.  
*Test:*
- AURG difference (Hybrid vs Sobel): Mean $\bar{\Delta} = +0.004692$, $t = 8.79$, $p = 2.96 \times 10^{-14}$, $94$ wins.
- 10% Sparsification difference: Mean $\bar{\Delta} = +1.59\%$, $t = 8.07$, $90$ wins.
- 20% Sparsification difference: Mean $\bar{\Delta} = +2.84\%$, $t = 9.69$, $98$ wins.
*Conclusion:* **REFUTED.** None of the secondary endpoints match $+0.0091$ or $t = 3.52$.

### Hypothesis 3: Component Comparison Misattribution (Refuted)
*Hypothesis:* The numbers represent Hybrid vs Input-Sensitivity, or Sobel vs Random.  
*Test:*
- Hybrid vs Input Sensitivity on holdout: Mean $\bar{\Delta} = +0.004538$, $t = 1.3664$, $p = 0.1747$, $W = 1783.0$ ($p = 0.000587$), $71$ wins / $36$ losses ($66.36\%$).
- Notice that Hybrid vs Input Sensitivity has $71$ wins ($66.36\%$). While $71$ is superficially close to $70$ ($65.42\%$), its $t$-statistic is $1.37$ ($p=0.17$), not $3.52$ ($p=0.0006$).
*Conclusion:* **REFUTED.**

### Hypothesis 4: Exhaustive Matrix Search across All Repository Data (Refuted)
*Hypothesis:* The numbers exist in an alternative development fold, pilot test, or historical run.  
*Test:* We executed an automated combinatoric search across all pairwise combinations of numeric columns in `per_tile_holdout_metrics.csv` ($44$ columns, $946$ pairs) and `per_tile_oof_metrics.csv` ($37$ columns, $666$ pairs).  
*Result:* Not a single column pair in any CSV yielded $m = 0.0091, t = 3.52, p = 0.0006, \text{wins } 70/107$. Furthermore, a text search across all workspace files for the strings `"65.42"` and `"70/107"` returned zero occurrences.  
*Conclusion:* **REFUTED.**

### Hypothesis 5: Conversational Generation Hallucination / Transcription Error (CONFIRMED)
*Finding:* In the conversational message generated in the preceding turn (EXP-020A synthesis), the LLM synthesized text for Phase 3:
> *"Statistically significant advantage over the Sobel gradient baseline: paired $\bar{\Delta} = +0.0091$ in rank correlation ($t = 3.52, p = 0.0006$, Wilcoxon $p = 0.0017$, win rate $65.42\%$ (70/107))."*

This text was generated dynamically in chat memory and did **NOT** originate from any disk file, code output, or validated table. The true on-disk holdout results are preserved without contradiction across all formal reports.

---

## 4. Specific Audit Mandate Inquiries

1. **Do the two sets of numbers correspond to different metrics, data, transcription error, or implementation change?**  
   **Answer:** The discrepancy is an isolated **conversational transcription hallucination** in the LLM's chat summary. It does not reflect any alternative metric, underlying data change, or script modification.
2. **Which metric was prespecified as the primary locked-holdout endpoint?**  
   **Answer:** The primary locked-holdout endpoint prespecified in EXP-013, EXP-015, EXP-016, and `EXP-017/preflight_protocol.json` is the **paired per-tile difference in Spearman rank correlation between the Frozen Hybrid Estimator and the Sobel Gradient Baseline on compensated interior crops**.
3. **Did EXP-018 independently recompute the exact same metric used in EXP-017's primary analysis?**  
   **Answer:** **Yes.** EXP-018 recomputed the exact primary metric directly from `per_tile_holdout_metrics.csv` and confirmed $\bar{\Delta} = +0.079316, t = 8.6251, p = 6.87 \times 10^{-14}, W = 458.0, 93/14$ wins.
4. **Did the synthesis accurately label and describe each result?**  
   **Answer:** **No.** The synthesis mistakenly inserted fabricated statistics for the holdout comparison in place of the verified EXP-017/018/019 values.
5. **Does the discrepancy affect the scientific conclusion or require an erratum?**  
   **Answer:** It does **not** undermine the scientific conclusion; in fact, the true empirical evidence on disk ($\bar{\Delta} = +0.0793, t = 8.63, p < 10^{-13}, 93/107$ wins) is substantially stronger and more robust than the erroneous chat text ($+0.0091, t=3.52, 70/107$ wins). An explicit erratum documenting this chat transcription error is formally established below.

---

## 5. Formal Erratum

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FORMAL AUDIT ERRATUM ERR-021-01                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Target Statement: Conversational Synthesis Turn (EXP-020A Conclusion, Section 2)        │
│ Erroneous Text:   "paired Δ = +0.0091 in rank correlation (t = 3.52, p = 0.0006,       │
│                    Wilcoxon p = 0.0017, win rate 65.42% (70/107))"                      │
│                                                                                        │
│ Corrected Text:   "paired Δ = +0.079316 in rank correlation (t = 8.6251, p = 6.87e-14, │
│                    Wilcoxon p = 4.17e-14, win rate 86.92% (93/107 wins))"               │
│                                                                                        │
│ Ground Truth:     results/satlas_sr_experiment_017_locked_holdout/summary.json:L88     │
│                   results/satlas_sr_experiment_018_holdout_consistency_audit/           │
│                   results/satlas_sr_experiment_019_research_report/EXP019_RESULTS_      │
│                   TABLES.csv:L15                                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
