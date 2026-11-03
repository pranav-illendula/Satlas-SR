# Satlas-SR Project: Executive Supervisor Handover Document

**Document Title**: Research Reproduction Handover: Satlas-SR Multi-Frame Super-Resolution Baseline  
**Investigation Cycle**: Baseline Freeze v1 & Experiments EXP-001 through EXP-010  
**Prepared For**: Research Supervisor / Academic Evaluation Committee  
**Prepared By**: Antigravity Autonomous Research & Verification System  
**Date**: October 22, 2026  
**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Git Branch**: `experiments` (HEAD Commit: `3e67973f6746bedbd91cc032b4888574980e1daf`)  
**Computing Environment**: Python 3.9.6 (.venv), PyTorch 2.1.0+cu121, CUDA 12.1, NVIDIA RTX 2000 Ada Generation (16 GB VRAM)  

---

## 1. Research Objective

The objective of this investigation was to conduct an evidence-based reproduction and critical audit of the baseline multi-frame satellite super-resolution model described in *Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing* (Wolters et al., arXiv:2311.18082v1).

The project rigorously evaluated the released open-source multi-frame ESRGAN architecture, standardized evaluation metrics against published definitions, established empirical sensitivity bounds, audited asset provenance, and identified the exact structural boundaries between the local reproducible baseline and the published paper.

---

## 2. Verified Experimental Setup

| Component | Specification | Authoritative Evidence Reference |
| :--- | :--- | :--- |
| **Model Architecture** | `SSR_RRDBNet` (Real-ESRGAN generator variant) | [`ssr/archs/rrdbnet_arch.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/archs/rrdbnet_arch.py) |
| **Model Capacity** | 23 RRDB blocks, 64 features, growth channel 32 | **`16,710,083` parameters** (702 weight/bias tensors) |
| **Model Size Tier** | **Small (16.71M parameters)** | Distinct from paper Table 2 Medium model (~87M) |
| **Official Checkpoint** | `experiments/esrgan_8S2.pth` | Size: **`134,157,518` bytes** (~127.94 MiB) |
| **Checkpoint SHA-256** | `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07` | Verified on disk and in runtime logs |
| **Input Sensor & Data** | Sentinel-2 L2A BOA Multi-Spectral ($10\text{ m/px}$) | 8 temporal revisits, 9 spectral bands ($72 \times 128 \times 128$) |
| **Ground Truth Reference** | USDA NAIP Aerial Imagery ($0.5\text{ m/px}$) | High-resolution RGB ($3 \times 512 \times 512$) |
| **Benchmark Dataset** | `data/small_val_set` | **256 paired tiles** (512 files, 100% paired, urban US) |

---

## 3. Verified Baseline Results (256 Urban Tiles)

The local baseline was evaluated deterministically across all $N=256$ validation tiles. Every number below was independently verified in EXP-009 directly from per-tile records within float tolerance ($\epsilon \le 10^{-4}$):

| Metric Name | Direction | Mean | Std Dev | Min | Max | Faithful Paper Table 2 Comparison? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **cPSNR (crop border 4)** | $\uparrow$ | **20.2144 dB** | 4.1478 dB | 9.5912 dB | 39.5373 dB | **No** (Local: 16.71M Small; Paper: 87M Medium) |
| **Raw Full-Frame PSNR** | $\uparrow$ | **18.8122 dB** | 3.8138 dB | 5.7260 dB | 33.8205 dB | **No** (Unaligned pixel MSE) |
| **Crop-Border-4 PSNR** | $\uparrow$ | **18.8870 dB** | 3.8600 dB | 5.5999 dB | 34.7798 dB | **No** (Boundary-excluded MSE) |
| **Raw Full-Frame SSIM** | $\uparrow$ | **0.3950** | 0.1495 | 0.0570 | 0.9059 | **No** (Standard structural similarity) |
| **Crop-Border-4 SSIM** | $\uparrow$ | **0.3912** | 0.1564 | 0.0501 | 0.9129 | **No** (Boundary-excluded SSIM) |
| **VGG LPIPS Distance** | $\downarrow$ | **0.3470** | 0.0809 | 0.1442 | 0.7196 | **No** (Raw distance vs. 63.8% MTurk win-rate) |
| **CLIPScore (Cosine Sim)** | $\uparrow$ | **0.9612** | 0.0344 | 0.7768 | 0.9930 | **No** (Continuous cosine in $[-1, 1]$) |

---

## 4. Summary of Completed Experiments (EXP-001 through EXP-010)

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT EXPERIMENT TIMELINE                                     |
+----------+--------------------------------------+--------------------------------------------------+
| Exp ID   | Focus Area                           | Core Finding / Verified Contribution             |
+----------+--------------------------------------+--------------------------------------------------+
| EXP-001  | Dense vs. Stride-2 cPSNR Sensitivity | Dense search yields +0.1232 dB mean gain         |
| EXP-002  | Paper Reproduction Gap Analysis      | 10-dimension gap matrix (Small vs. Medium model) |
| EXP-003  | Metric Standardization & LPIPS       | Refuted 1 - LPIPS; preserved raw VGG distance    |
| EXP-004  | Provenance & Asset Audit             | Traced commit 7ccf8d1; confirmed unreleased 87M  |
| EXP-005  | Supplementary PDF Verification       | Inspected Section A.1; confirmed 11,524 MTurk AMT|
| EXP-006  | Final Baseline Reproducibility Audit | Consolidated evidence; noted text transcription  |
| EXP-007  | Checkpoint Reconciliation            | Proved physical checkpoint is 134MB (167d6b9d...) |
| EXP-008  | Research Report Consolidation        | Produced academic report & 3 SVG vector figures  |
| EXP-009  | Independent Numerical Verification   | 49/50 checks passed; audited tolerances (e<1e-4) |
| EXP-010  | Handover & Git Erratum (Current)     | Reconciled commit 3e67973; supervisor handover   |
+----------+--------------------------------------+--------------------------------------------------+
```

- **EXP-001 (cPSNR Grid Search Sensitivity)**: Proved dense stride-1 search space is a mathematical superset of stride-2 search, guaranteeing non-negative gain ($+0.1232\text{ dB}$, 95% CI: $[0.1018, 0.1477]\text{ dB}$). Stride-2 remains the frozen baseline standard.
- **EXP-002 (Paper Gap Analysis)**: Formulated a 10-dimension matrix proving local baseline evaluates the Small model on 256 urban tiles, whereas published Table 2 evaluates the Medium model on 8,192 multi-biome tiles.
- **EXP-003 (Metric Standardization)**: Disproved the naive $1 - \text{LPIPS} = 0.6530$ formula and standardized reporting on raw VGG LPIPS distance ($0.3470$).
- **EXP-004 (Provenance Audit)**: Traced Git commit `7ccf8d1` migrating assets to GCS and verified that the 87M Medium ESRGAN checkpoint was never publicly released.
- **EXP-005 (Supplementary Material Inspection)**: Downloaded and verified `Zooming_Out_On_Zooming_In_Supplementary.pdf` (1.74 MB), confirming that Table 2 LPIPS accuracy represents a crowdsourced 2AFC human Mechanical Turk study with no mathematical conversion formula.
- **EXP-006 (Consolidation Audit)**: Synthesized local baseline reproducibility across all artifacts.
- **EXP-007 (Checkpoint Reconciliation)**: Resolved report transcription errors, mathematically proving the active checkpoint is $134,157,518\text{ bytes}$ with SHA-256 `167d6b9d...`, strictly matching the original inference runs.
- **EXP-008 (Consolidated Research Package)**: Produced `FINAL_RESEARCH_REPORT.md`, `NUMERICAL_RECONCILIATION.md`, `SOURCE_AND_ARTIFACT_INDEX.md`, `final_report_data.json`, and 3 vector SVG figures.
- **EXP-009 (Independent Numerical Verification)**: Executed 50 automated tests; 49 passed, with 1 failure isolating a Git commit header erratum.
- **EXP-010 (Final Handover & Git Erratum)**: Produced this handover document, `GIT_METADATA_ERRATUM.md`, and `FINAL_ARTIFACT_INDEX.md`.

---

## 5. Reproducibility Evidence and Audit Trail

The baseline is backed by an unbroken cryptographic and procedural audit trail:
1. **Frozen Baseline Package (`v1`)**: Immutable snapshot in [`results/satlas_sr_baseline_freeze_v1/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1).
2. **Runtime Ingestion Logging**: In [`results/satlas_sr_full_validation/inference_manifest.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv), line-by-line runtime hashing logged `167d6b9d...` dynamically for all 256 tiles during inference.
3. **Independent Recalculation**: In EXP-009, 100% of numeric metrics were recomputed from raw CSVs within float tolerance ($\le 10^{-4}$).
4. **Git Metadata Discrepancy Resolved**: The single failure in EXP-009 (`CHK-GIT-exp008-meta`) was investigated in EXP-010. The true repository commit is **`3e67973f6746bedbd91cc032b4888574980e1daf`** (matching `baseline_manifest.json` line 9), and the erratum string `eb3dcae4...` has been formally documented in [`results/satlas_sr_experiment_010_final_handover/GIT_METADATA_ERRATUM.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_010_final_handover/GIT_METADATA_ERRATUM.md).

---

## 6. Critical Scientific Limitations

1. **Model Capacity Disparity**: The local baseline evaluates the **16.71M Small model**, whereas the published paper's Table 2 evaluated an **87M Medium model**. These capacities cannot be directly equated.
2. **Dataset Distribution Mismatch**: The local baseline evaluates 256 urban tiles from CONUS. The published paper evaluated an 8,192-tile benchmark stratified across urban, rural, forest, and arid biomes.
3. **LPIPS Distance vs. Human Agreement**: The local metric ($0.3470$) is raw perceptual feature distance. Table 2 reports an empirical win-rate ($63.8\%$) from 11,524 crowd-worker judgments. No scalar formula maps one to the other.
4. **CLIPScore Interpretation**: The score of $0.9612$ is a continuous normalized vector cosine in $[-1, 1]$, not a percentage accuracy.
5. **Spatial Registration Diagnostic**: Fourier phase correlation ($3.863 \pm 5.741\text{ px}$) diagnoses relative high-frequency pattern alignment. It does **not** prove geodetic sensor alignment or the absence of geometric hallucination.

---

## 7. Outstanding Work & Upstream Gaps

```
+----------------------------------------------------------------------------------------------------+
|                                    REPRODUCTION STATUS SUMMARY                                     |
+--------------------------+---------------------+---------------------------------------------------+
| Level                    | Reproduction Status | Current Operational Status                        |
+--------------------------+---------------------+---------------------------------------------------+
| Level 1: Local Baseline  | FULLY REPRODUCED    | 100% complete, frozen, auditable, and verified    |
| Level 2: Metric Suite    | FULLY VALIDATED     | Algorithms verified against author implementations|
| Level 3: Paper Table 2   | BLOCKED UPSTREAM    | Blocked by unreleased 87M weights & MTurk data    |
+--------------------------+---------------------+---------------------------------------------------+
```

### Unreleased Upstream Resources (Outside Local Control):
- The trained weights for the 87M Medium ESRGAN model were never distributed.
- The 11,524 Amazon Mechanical Turk human preference comparison records were not published.

### Potential Future Local Work (If Resources Allocated):
- **Cross-Biome Dataset Expansion**: Download the full 8,192 validation tiles from Google Cloud Storage (`data/val/`) to benchmark non-urban biomes.
- **Local Medium Model Training**: Train an 87M Medium ESRGAN from scratch using `ssr/train.py` if multi-GPU high-performance computing resources are provisioned.

---

## 8. Recommended Next Steps

1. **Supervisor Review (Immediate)**: Review this handover document and the consolidated [`FINAL_RESEARCH_REPORT.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_008_final_research_report/FINAL_RESEARCH_REPORT.md).
2. **Maintain Freeze on Existing Assets**: Retain `results/satlas_sr_baseline_freeze_v1/`, `results/satlas_sr_experiment_008_final_research_report/`, `results/satlas_sr_experiment_009_independent_verification/`, and `results/satlas_sr_experiment_010_final_handover/` as immutable historical records.
3. **Decision on Future Directions**:
   - *Path A (Baseline Complete)*: Conclude reproduction at Level 1/Level 2, adopting the 16.71M model on 256 tiles as the benchmark for subsequent algorithmic modifications.
   - *Path B (Resource Acquisition)*: Contact AllenAI authors to formally request the unreleased 87M weights and MTurk comparison annotations if exact Table 2 reproduction is mandatory.
   - *Path C (Retraining)*: Initiate local multi-GPU training of the larger architecture.
