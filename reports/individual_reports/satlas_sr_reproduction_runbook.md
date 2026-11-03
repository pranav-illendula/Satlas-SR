# Satlas-SR Reproduction Runbook & Operational Guide

This runbook documents exact, verified commands to inspect data, run inference, evaluate metrics, audit spatial registration, perform statistical analysis, and generate reports on Windows and POSIX systems.

---

## 1. Environment & Setup

Ensure the virtual environment is activated and dependencies are present:

```powershell
# Windows PowerShell
$PYTHON = "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe"
cd C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution

# Verify CUDA and GPU accessibility
& $PYTHON -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

## 2. Dataset & Checkpoint Inventory Audit

Audits the 256 tiles, verifies 3-way pairing (S2 `tci.png`, NAIP `rgb.png`, SR prediction), checks PNG corruption, and verifies model parameter count:

```powershell
& $PYTHON scripts/audit_dataset_and_inference.py
```
- **Outputs**:
  - `results/satlas_sr_reproduction_audit/dataset_inventory.csv`
  - `results/satlas_sr_reproduction_audit/dataset_inventory.json`
  - `results/satlas_sr_reproduction_audit/dataset_validation_report.md`

---

## 3. Path Compatibility & Unit Tests

Executes focused unit tests for cross-platform path compatibility:

```powershell
& $PYTHON tests/test_dataset_path_compatibility.py
```

---

## 4. Full Validation Inference (Deterministic)

*Note: Since the 256-tile inference is already completed and verified under `results/satlas_sr_full_validation/`, re-running this step will safely skip already-computed valid tiles.*

```powershell
& $PYTHON run_full_validation_inference.py -opt ssr/options/infer_satlas_full_validation.yml
```
- **Outputs**:
  - `results/satlas_sr_full_validation/{tile_id}/sr.png` (Super-resolved prediction, 128x128)
  - `results/satlas_sr_full_validation/{tile_id}/lr.png` (Low-resolution input, 32x32)
  - `results/satlas_sr_full_validation/inference_manifest.csv`

---

## 5. Metric Revalidation & Deep-Dive Verification

Recomputes official paper metrics across all 256 tiles from disk, verifying cPSNR, LPIPS, CLIPScore, PSNR, and SSIM:

```powershell
# 1. Full 256-tile revalidation
& $PYTHON scripts/revalidate_and_compare_metrics.py

# 2. Mathematical deep-dive audit (cPSNR stride, LPIPS normalization, CLIPScore sensitivity)
& $PYTHON scripts/verify_metrics_deep_dive.py
```
- **Outputs**:
  - `results/satlas_sr_revalidation/per_tile_metrics.csv`
  - `results/satlas_sr_revalidation/metric_comparison.csv`
  - `results/satlas_sr_revalidation/aggregate_metrics.json`
  - `results/satlas_sr_reproduction_audit/metric_verification_deep_dive.json`

---

## 6. Spatial Registration Diagnostic & Reassessment

Calibrates synthetic shift estimators, filters spectral wraparound artifacts, cross-verifies with Normalized Cross-Correlation (NCC), and generates filtered statistics:

```powershell
& $PYTHON scripts/diagnose_registration_estimator.py
```
- **Outputs**:
  - `results/satlas_sr_spatial_audit/registration_validation.csv`
  - `results/satlas_sr_spatial_audit/registration_validation_report.md`
  - `reports/satlas_sr_spatial_registration_reassessment.md`

---

## 7. Statistical & Failure Analysis

Computes 2,000-resample bootstrap 95% confidence intervals, Pearson & Spearman correlation matrices, ranked extreme failure tables, and generates visual contact sheets:

```powershell
& $PYTHON scripts/run_statistical_and_failure_analysis.py
```
- **Outputs**:
  - `results/satlas_sr_revalidation/statistical_analysis.json`
  - `results/satlas_sr_revalidation/metric_correlations.csv`
  - `results/satlas_sr_revalidation/visualizations_v2/` (Labeled PNG contact sheets)
  - `reports/satlas_sr_statistical_analysis.md`
  - `reports/satlas_sr_per_tile_failure_analysis.md`

---

## 8. Consolidated Master Reporting

Consolidated reports are available under `reports/`:
1. `reports/SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md`: Master synthesis of all findings.
2. `reports/satlas_sr_paper_comparability.md`: Direct comparison against paper claims and benchmark scope.
3. `reports/satlas_sr_model_fidelity_audit.md`: Architecture and parameter verification.
4. `reports/satlas_sr_windows_compatibility_audit.md`: Cross-platform path handling.
5. `reports/satlas_sr_metric_validation_deep_dive.md`: Deep dive on metric mechanics.
