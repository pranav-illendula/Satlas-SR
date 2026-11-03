# Satlas-SR Project: Master Artifact and Verification Index

**Document Purpose**: Definitive, verified catalog of all persistent artifacts, code modules, datasets, checkpoints, and experiment deliverables generated across the Satlas-SR multi-frame super-resolution reproduction project.

Every file listed in this index has been verified for existence, byte size, and read-only integrity.

---

## 1. Frozen Baseline & Primary Pipeline

| Category | Relative / Absolute Path | Size / Integrity | Purpose & Scope |
| :--- | :--- | :---: | :--- |
| **Model Weights** | [`experiments/esrgan_8S2.pth`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/experiments/esrgan_8S2.pth) | 134,157,518 bytes<br>SHA-256: `167d6b9d...` | Official AllenAI small multi-frame ESRGAN generator (16.71M parameters). |
| **Validation Dataset** | [`C:/Users/CSE-SDPL/Desktop/DL_image/data/small_val_set`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/data/small_val_set) | 512 files<br>(256 inputs, 256 targets) | Multi-temporal Sentinel-2 low-resolution PNGs paired with 0.5m NAIP aerial RGB targets. |
| **Frozen Manifest** | [`results/satlas_sr_baseline_freeze_v1/baseline_manifest.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1/baseline_manifest.json) | 8,462 bytes | Authoritative baseline v1 freeze manifest defining commit, environment, and weights. |
| **Freeze Report** | [`results/satlas_sr_baseline_freeze_v1/BASELINE_FREEZE_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1/BASELINE_FREEZE_REPORT.md) | 14,095 bytes | Baseline freeze validation report and integrity certification. |
| **Reproduction Runbook**| [`results/satlas_sr_baseline_freeze_v1/REPRODUCTION_RUNBOOK.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1/REPRODUCTION_RUNBOOK.md) | 6,928 bytes | Baseline v1 execution guide with output protection rules. |
| **Experiment Template** | [`results/satlas_sr_baseline_freeze_v1/EXPERIMENT_TEMPLATE.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1/EXPERIMENT_TEMPLATE.md) | 3,369 bytes | Standardized research experiment protocol. |

---

## 2. Full-Validation Inference & Paper Metrics

| Artifact Name | Path | Size | Description |
| :--- | :--- | :---: | :--- |
| **Inference Manifest** | [`results/satlas_sr_full_validation/inference_manifest.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv) | 199,322 bytes | Runtime execution log for all 256 tiles recording per-tile SHA-256 (`167d6b9d...`). |
| **Inference Summary** | [`results/satlas_sr_full_validation/run_summary.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/run_summary.md) | 1,912 bytes | Summary of 256 completed super-resolution predictions. |
| **Per-Tile Metrics** | [`results/satlas_sr_paper_metrics/per_tile_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_paper_metrics/per_tile_metrics.csv) | 48,162 bytes | Authoritative per-tile evaluations across cPSNR, PSNR, SSIM, LPIPS, and CLIPScore. |
| **Metrics Summary JSON**| [`results/satlas_sr_paper_metrics/summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_paper_metrics/summary.json) | 6,055 bytes | Machine-readable aggregate evaluation statistics. |
| **Evaluation Report** | [`results/satlas_sr_paper_metrics/evaluation_report.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_paper_metrics/evaluation_report.md) | 6,036 bytes | Official paper-metrics baseline report. |

---

## 3. Specialized Research Experiments (EXP-001 through EXP-007)

### EXP-001: Dense vs. Stride-2 cPSNR Sensitivity
- Report: [`results/satlas_sr_experiment_001/EXPERIMENT_001_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_001/EXPERIMENT_001_REPORT.md) (16,071 bytes)
- Summary JSON: [`results/satlas_sr_experiment_001/experiment_001_summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_001/experiment_001_summary.json) (1,746 bytes)
- Per-Tile CSV: [`results/satlas_sr_experiment_001/dense_cpsnr_per_tile.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_001/dense_cpsnr_per_tile.csv) (33,485 bytes)
- Finding: Dense search yields $+0.1232\text{ dB}$ mean gain over stride-2 search across 256 tiles.

### EXP-002: Paper-Level Reproduction Gap Analysis
- Gap Matrix: [`results/satlas_sr_experiment_002_paper_gap_analysis/PAPER_REPRODUCTION_GAP_MATRIX.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_002_paper_gap_analysis/PAPER_REPRODUCTION_GAP_MATRIX.md) (12,607 bytes)
- Roadmap: [`results/satlas_sr_experiment_002_paper_gap_analysis/REPRODUCTION_ROADMAP.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_002_paper_gap_analysis/REPRODUCTION_ROADMAP.md) (9,536 bytes)
- Inventory: [`results/satlas_sr_experiment_002_paper_gap_analysis/paper_evidence_inventory.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_002_paper_gap_analysis/paper_evidence_inventory.json) (5,600 bytes)
- Finding: Formal 10-dimension gap analysis delineating Small vs Medium model capacity and dataset scopes.

### EXP-003: Metric Standardization & LPIPS Investigation
- Report: [`results/satlas_sr_experiment_003_metric_standardization/METRIC_STANDARDIZATION_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_003_metric_standardization/METRIC_STANDARDIZATION_REPORT.md) (13,124 bytes)
- Summary JSON: [`results/satlas_sr_experiment_003_metric_standardization/metric_standardization_summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_003_metric_standardization/metric_standardization_summary.json) (2,778 bytes)
- Per-Tile CSV: [`results/satlas_sr_experiment_003_metric_standardization/lpips_agreement_per_tile.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_003_metric_standardization/lpips_agreement_per_tile.csv) (15,920 bytes)
- Finding: Refuted naive $1 - \text{LPIPS}$ conversion; established raw perceptual distance ($0.3470$).

### EXP-004: Checkpoint Provenance & Author Repository Audit
- Report: [`results/satlas_sr_experiment_004_provenance_audit/PROVENANCE_AUDIT_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_004_provenance_audit/PROVENANCE_AUDIT_REPORT.md) (12,854 bytes)
- Inventory: [`results/satlas_sr_experiment_004_provenance_audit/provenance_inventory.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_004_provenance_audit/provenance_inventory.json) (5,595 bytes)
- Source Evidence: [`results/satlas_sr_experiment_004_provenance_audit/source_evidence.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_004_provenance_audit/source_evidence.csv) (4,524 bytes)
- Finding: Discovered author commit `7ccf8d1` migrating to GCS; confirmed 87M Medium checkpoint was unreleased.

### EXP-005: Authoritative Supplementary PDF Verification
- Report: [`results/satlas_sr_experiment_005_supplementary_verification/SUPPLEMENTARY_VERIFICATION_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_005_supplementary_verification/SUPPLEMENTARY_VERIFICATION_REPORT.md) (9,252 bytes)
- Inventory: [`results/satlas_sr_experiment_005_supplementary_verification/source_inventory.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_005_supplementary_verification/source_inventory.json) (4,680 bytes)
- Downloaded PDF: [`results/satlas_sr_experiment_005_supplementary_verification/Zooming_Out_On_Zooming_In_Supplementary.pdf`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_005_supplementary_verification/Zooming_Out_On_Zooming_In_Supplementary.pdf) (1,821,393 bytes, SHA-256 `ebdf117d...`)
- Finding: Supplementary Section A.1 specifies a 2AFC human AMT study, confirming no scalar formula exists.

### EXP-006: Final Baseline Reproducibility and Reporting Audit
- Report: [`results/satlas_sr_experiment_006_final_audit/FINAL_REPRODUCTION_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit/FINAL_REPRODUCTION_REPORT.md) (12,735 bytes)
- Runbook: [`results/satlas_sr_experiment_006_final_audit/REPRODUCIBILITY_RUNBOOK.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit/REPRODUCIBILITY_RUNBOOK.md) (5,419 bytes)
- Inventory: [`results/satlas_sr_experiment_006_final_audit/audit_inventory.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit/audit_inventory.json) (7,469 bytes)
- Integrity CSV: [`results/satlas_sr_experiment_006_final_audit/artifact_integrity_check.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit/artifact_integrity_check.csv) (4,296 bytes)

### EXP-007: Checkpoint Provenance Reconciliation
- Report: [`results/satlas_sr_experiment_007_checkpoint_reconciliation/CHECKPOINT_RECONCILIATION_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_007_checkpoint_reconciliation/CHECKPOINT_RECONCILIATION_REPORT.md) (9,052 bytes)
- Inventory: [`results/satlas_sr_experiment_007_checkpoint_reconciliation/checkpoint_inventory.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_007_checkpoint_reconciliation/checkpoint_inventory.json) (5,104 bytes)
- Reference Search: [`results/satlas_sr_experiment_007_checkpoint_reconciliation/checkpoint_reference_search.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_007_checkpoint_reconciliation/checkpoint_reference_search.csv) (105,460 bytes)
- Finding: Formally resolved EXP-006 drafting error. Proved active checkpoint is $134,157,518\text{ bytes}$ with SHA-256 `167d6b9d...`.

---

## 4. Final Consolidated Research Package (EXP-008, EXP-009, EXP-010)

### EXP-008: Final Research Report Consolidation
- Main Report: [`results/satlas_sr_experiment_008_final_research_report/FINAL_RESEARCH_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/FINAL_RESEARCH_REPORT.md) (20,573 bytes)
- Reconciliation: [`results/satlas_sr_experiment_008_final_research_report/NUMERICAL_RECONCILIATION.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/NUMERICAL_RECONCILIATION.md) (10,001 bytes)
- Artifact Index: [`results/satlas_sr_experiment_008_final_research_report/SOURCE_AND_ARTIFACT_INDEX.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/SOURCE_AND_ARTIFACT_INDEX.md) (7,759 bytes)
- Data JSON: [`results/satlas_sr_experiment_008_final_research_report/final_report_data.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/final_report_data.json) (7,916 bytes)
- Figure 1: [`results/satlas_sr_experiment_008_final_research_report/fig1_cpsnr_distribution.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/fig1_cpsnr_distribution.svg) (7,997 bytes)
- Figure 2: [`results/satlas_sr_experiment_008_final_research_report/fig2_dense_cpsnr_gain.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/fig2_dense_cpsnr_gain.svg) (7,110 bytes)
- Figure 3: [`results/satlas_sr_experiment_008_final_research_report/fig3_metric_profiles.svg`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/fig3_metric_profiles.svg) (2,668 bytes)

### EXP-009: Independent Numerical Verification
- Report: [`results/satlas_sr_experiment_009_independent_verification/INDEPENDENT_VERIFICATION_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_009_independent_verification/INDEPENDENT_VERIFICATION_REPORT.md) (11,341 bytes)
- Results JSON: [`results/satlas_sr_experiment_009_independent_verification/verification_results.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_009_independent_verification/verification_results.json) (18,358 bytes)
- Command Record: [`results/satlas_sr_experiment_009_independent_verification/verification_command_record.txt`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_009_independent_verification/verification_command_record.txt) (3,577 bytes)
- Verdict: 49/50 checks passed; recomputed all metrics within float tolerance $\le 10^{-4}$.

### EXP-010: Handover & Git Erratum (Current)
- Git Erratum: [`results/satlas_sr_experiment_010_final_handover/GIT_METADATA_ERRATUM.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_010_final_handover/GIT_METADATA_ERRATUM.md)
- Supervisor Handover: [`results/satlas_sr_experiment_010_final_handover/SUPERVISOR_HANDOVER.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_010_final_handover/SUPERVISOR_HANDOVER.md)
- Master Artifact Index: [`results/satlas_sr_experiment_010_final_handover/FINAL_ARTIFACT_INDEX.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_010_final_handover/FINAL_ARTIFACT_INDEX.md)
