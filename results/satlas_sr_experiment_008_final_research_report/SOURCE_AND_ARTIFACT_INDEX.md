# Satlas-SR EXP-008: Source Code, Data, and Artifact Index

This index provides a complete, traceable catalog of all software source code, configurations, datasets, model checkpoints, and experimental artifacts across the entire Satlas-SR research reproduction project.

---

## 1. Primary Repository and Codebase

| Resource | Path / Identifier | Verified Status | Description |
| :--- | :--- | :---: | :--- |
| **Repository Root** | `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution` | Active | Cloned official AllenAI Satlas-SR repository |
| **Git Commit (Head)** | `eb3dcae4cae4492652b04c86cb31d167f8b9d997` | Verified | Base commit for evaluation pipeline |
| **Working Branch** | `master` / tracked modifications | Preserved | Minimal compatibility modifications for Windows path separators and CLIP import |
| **Python Environment** | `C:\Users\CSE-SDPL\Desktop\DL_image\.venv` | Python 3.9.6 | Dedicated virtual environment with PyTorch 2.1.0+cu121 |

### Key Source Code Modules:
- [`ssr/archs/rrdbnet_arch.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/archs/rrdbnet_arch.py): Defines `SSR_RRDBNet` multi-frame super-resolution generator (16.71M parameters).
- [`ssr/metrics/cpsnr.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/metrics/cpsnr.py): Official PROBA-V style stride-2 translation search and brightness-bias compensation.
- [`ssr/metrics/clipscore.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/metrics/clipscore.py): CLIPScore evaluation using OpenCLIP `clipa-ViT-bigG-14-datacomp1b`.
- [`ssr/data/s2-naip_dataset.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/data/s2-naip_dataset.py): Multi-temporal Sentinel-2 to high-resolution NAIP dataset loader.
- [`run_full_validation_inference.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/run_full_validation_inference.py): Resumable inference script with dynamic per-tile SHA-256 runtime hashing.

---

## 2. Model Checkpoint Provenance

| Property | Value | Verification Reference |
| :--- | :--- | :--- |
| **Checkpoint Path** | `experiments/esrgan_8S2.pth` | Active on disk |
| **File Size** | **`134,157,518` bytes** | Certified in EXP-007 |
| **SHA-256 Hash** | **`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`** | Match with runtime inference logs |
| **File Timestamp** | `2026-10-15 15:44:59.790044` | Unmodified since initial download |
| **Download Source** | `https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/esrgan_8S2.pth` | Author README URL |
| **GCS Mirror URL** | `https://storage.googleapis.com/satlas-satellite-super-resolution/esrgan_8S2.pth` | Commit `7ccf8d1` mirror |
| **State Dictionaries** | `['params', 'params_ema']` | 702 tensors, 16,710,083 parameters each |
| **Loaded Key** | `params_ema` | Specified in `ssr/options/infer_satlas_full_validation.yml` |

---

## 3. Dataset Inventory (`data/small_val_set`)

| Directory | Count | Specification | Integrity |
| :--- | :---: | :--- | :---: |
| **Sentinel-2 Inputs** | 256 tiles | Stacked temporal PNGs ($32 \times 32 \times 3$, 32 frames), 9 spectral bands | 100% Valid |
| **NAIP Targets** | 256 tiles | High-resolution aerial RGB ($512 \times 512$, 0.5m GSD) | 100% Valid |
| **Tile Pairing** | 256 / 256 | Exact geographic and tile ID correspondence | Zero Missing |

---

## 4. Experiment Directory Catalog

### Baseline Freeze v1
- Directory: [`results/satlas_sr_baseline_freeze_v1/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1)
- Key Files: `baseline_manifest.json`, `BASELINE_FREEZE_REPORT.md`, `REPRODUCTION_RUNBOOK.md`, `EXPERIMENT_TEMPLATE.md`
- Scope: Permanent baseline freeze package of the 16.71M Small ESRGAN checkpoint on 256 tiles.

### EXP-001: Dense vs. Stride-2 cPSNR Sensitivity
- Directory: [`results/satlas_sr_experiment_001/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_001)
- Key Files: `EXPERIMENT_001_REPORT.md`, `experiment_001_summary.json`, `dense_cpsnr_per_tile.csv`
- Finding: Dense search produces mean gain of $+0.1232$ dB ($95\%$ CI: $[0.1018, 0.1477]$ dB) across 256 tiles.

### EXP-002: Paper-Level Reproduction Gap Analysis
- Directory: [`results/satlas_sr_experiment_002_paper_gap_analysis/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_002_paper_gap_analysis)
- Key Files: `GAP_ANALYSIS_REPORT.md`, `PAPER_REPRODUCTION_GAP_MATRIX.md`, `paper_evidence_inventory.json`
- Finding: 10-dimension gap matrix proving local baseline evaluates Small model on 256 urban tiles, whereas paper evaluated Medium model on 8,192 multi-biome tiles.

### EXP-003: Metric Standardization & LPIPS Investigation
- Directory: [`results/satlas_sr_experiment_003_metric_standardization/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_003_metric_standardization)
- Key Files: `METRIC_STANDARDIZATION_REPORT.md`, `metric_standardization_summary.json`, `lpips_agreement_per_tile.csv`
- Finding: Refuted naive $1 - \text{LPIPS}$ inversion. Preserved raw perceptual distance ($0.3470$).

### EXP-004: Checkpoint & Provenance Audit
- Directory: [`results/satlas_sr_experiment_004_provenance_audit/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_004_provenance_audit)
- Key Files: `PROVENANCE_AUDIT_REPORT.md`, `provenance_inventory.json`, `source_evidence.csv`
- Finding: Discovered author commit `7ccf8d1` migrating assets to GCS. Confirmed 87M medium checkpoint was never publicly released.

### EXP-005: Supplementary Material PDF Verification
- Directory: [`results/satlas_sr_experiment_005_supplementary_verification/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_005_supplementary_verification)
- Key Files: `SUPPLEMENTARY_VERIFICATION_REPORT.md`, `source_inventory.json`, `Zooming_Out_On_Zooming_In_Supplementary.pdf`
- Finding: Supplementary Section A.1 establishes Table 2 LPIPS accuracy is 2AFC human AMT win-rate from 11,524 trials, with no scalar conversion formula.

### EXP-006: Final Baseline Reproducibility and Reporting Audit
- Directory: [`results/satlas_sr_experiment_006_final_audit/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit)
- Key Files: `FINAL_REPRODUCTION_REPORT.md`, `REPRODUCIBILITY_RUNBOOK.md`, `audit_inventory.json`, `artifact_integrity_check.csv`
- Scope: Synthesized evidence across EXP-001 through EXP-005.

### EXP-007: Checkpoint Provenance Reconciliation
- Directory: [`results/satlas_sr_experiment_007_checkpoint_reconciliation/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_007_checkpoint_reconciliation)
- Key Files: `CHECKPOINT_RECONCILIATION_REPORT.md`, `checkpoint_inventory.json`, `checkpoint_reference_search.csv`
- Finding: Resolved EXP-006 reporting transcription discrepancy. Confirmed physical checkpoint is 134,157,518 bytes (`167d6b9d...`), matching baseline manifest and original inference runs.

### EXP-008: Final Satlas-SR Research Report Consolidation (Current)
- Directory: [`results/satlas_sr_experiment_008_final_research_report/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report)
- Deliverables:
  - `FINAL_RESEARCH_REPORT.md`
  - `NUMERICAL_RECONCILIATION.md`
  - `SOURCE_AND_ARTIFACT_INDEX.md`
  - `final_report_data.json`
  - `run_command_record.txt`
  - `fig1_cpsnr_distribution.svg`
  - `fig2_dense_cpsnr_gain.svg`
  - `fig3_metric_profiles.svg`
