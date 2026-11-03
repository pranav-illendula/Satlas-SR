# Satlas-SR Frozen Baseline Reproduction Runbook

- **Package Version**: `baseline_freeze_v1`
- **Date**: 2026-10-22
- **Environment**: Windows PowerShell, Python virtual environment (`C:\Users\CSE-SDPL\Desktop\DL_image\.venv`)
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`

This runbook documents exact, verified commands to inspect the environment, run unit tests, execute safe pilot inference, re-evaluate metrics from disk, and verify checkpoint integrity without rerunning full inference or modifying baseline files.

---

## 1. Environment Activation & Hardware Verification

Open Windows PowerShell and verify the environment:

```powershell
# Set Python interpreter variable
$PYTHON = "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe"
cd C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution

# Verify Python, PyTorch, CUDA, and GPU accessibility [EXECUTED & VERIFIED]
& $PYTHON -c "import torch; print('Python:', torch.__file__); print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / (1024**3))"
```
*Expected output*: CUDA Available: `True` | GPU: `NVIDIA RTX 2000 Ada Generation` | VRAM: ~`16.0 GB`.

---

## 2. Checkpoint Integrity & Parameter Count Verification

Verify the SHA-256 hash and parameter count of the official released checkpoint:

```powershell
# 1. Verify SHA-256 hash in PowerShell [EXECUTED & VERIFIED]
Get-FileHash -Algorithm SHA256 experiments\esrgan_8S2.pth | Select-Object Hash, Path

# 2. Verify file size and exact parameter count directly via PyTorch [EXECUTED & VERIFIED]
& $PYTHON -c "import hashlib, torch; from ssr.archs.rrdbnet_arch import SSR_RRDBNet; h = hashlib.sha256(open('experiments/esrgan_8S2.pth', 'rb').read()).hexdigest(); ckpt = torch.load('experiments/esrgan_8S2.pth', map_location='cpu'); m = SSR_RRDBNet(num_in_ch=24, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32); print('SHA-256:', h); print('Model parameters:', sum(p.numel() for p in m.parameters())); print('params_ema elements:', sum(v.numel() for v in ckpt['params_ema'].values()))"
```
*Expected output*:
- SHA-256: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`
- File Size: `134,157,518 bytes`
- Model parameters: `16,710,083`
- `params_ema` elements: `16,710,083`

---

## 3. Running Unit Tests

Run the unit test suites to verify metric mathematics and Windows path compatibility:

```powershell
# 1. Dataset path compatibility test (Windows backslash vs POSIX forward-slash) [EXECUTED & VERIFIED]
& $PYTHON tests/test_dataset_path_compatibility.py

# 2. Metric mathematical rigor tests (cPSNR, LPIPS, CLIPScore sensitivity) [EXECUTED & VERIFIED]
& $PYTHON tests/test_metrics_rigor.py
```
*Expected output*:
- `test_dataset_path_compatibility.py`: Ran 3 tests in 0.000s, `OK`.
- `test_metrics_rigor.py`: Ran 14 tests in ~74s, `OK`.

---

## 4. Safe Pilot Subset Inference (Optional)

To test the complete inference pipeline without touching or rerunning the 256-tile full validation set, execute the existing safe five-tile pilot or run a targeted subset to a dedicated test directory:

### Option A: Official 5-Tile Pilot Runner
Uses `run_five_tile_inference.py` (which targets `results/satlas_sr_pilot_five_tiles/`):
```powershell
# [INSPECTED ARGUMENTS: -opt <path>]
& $PYTHON run_five_tile_inference.py -opt ssr/options/infer_satlas_five_tiles.yml
```

### Option B: Targeted Subset Inference
Uses `run_full_validation_inference.py` with `--tiles` and `--output-dir` arguments:
```powershell
# [INSPECTED ARGUMENTS: -opt <path> --tiles <id1> <id2> --output-dir <path>]
# NOTE: Marked [NOT EXECUTED DURING FREEZE] to preserve state
& $PYTHON run_full_validation_inference.py -opt ssr/options/infer_satlas_full_validation.yml --tiles 20771_46525 20890_46801 --output-dir results/test_subset_smoke
```

---

## 5. Metric Evaluation from Disk (No Rerunning Inference)

Evaluate the existing 256 super-resolved predictions against their NAIP ground-truth references:

### Option A: Official Paper Metrics Runner
```powershell
# [INSPECTED ARGUMENTS: --predictions-dir <path> --naip-dir <path> --output-dir <path> --crop-border 4 --clip-model clipa-ViT-bigG-14]
# [EXECUTED & VERIFIED IN EARLIER AUDIT]
& $PYTHON evaluate_satlas_sr_paper_metrics.py --predictions-dir results/satlas_sr_full_validation --naip-dir ../data/small_val_set/naip --output-dir results/satlas_sr_paper_metrics --crop-border 4 --clip-model clipa-ViT-bigG-14
```

### Option B: Revalidation and Cross-Run Verification Script
```powershell
# [NO CLI ARGUMENTS REQUIRED; reads predictions and compares with prior run]
# [EXECUTED & VERIFIED]
& $PYTHON scripts/revalidate_and_compare_metrics.py
```
*Expected output*: Exact numerical reproduction across all 256 tiles ($|\Delta| < 10^{-3}$).

---

## 6. Spatial Registration Diagnostic Audit

Run the spatial registration diagnostic to evaluate phase cross-correlation, windowed phase correlation, and NCC:

```powershell
# [NO CLI ARGUMENTS REQUIRED]
# [EXECUTED & VERIFIED]
& $PYTHON scripts/diagnose_registration_estimator.py
```
*Outputs*:
- `results/satlas_sr_spatial_audit/registration_validation.csv`
- `results/satlas_sr_spatial_audit/registration_validation_report.md`
- `reports/satlas_sr_spatial_registration_reassessment.md`

---

## 7. Statistical & Failure Analysis

Recompute bootstrap confidence intervals, correlation matrices, and generate labeled failure contact sheets:

```powershell
# [NO CLI ARGUMENTS REQUIRED]
# [EXECUTED & VERIFIED]
& $PYTHON scripts/run_statistical_and_failure_analysis.py
```
*Outputs*:
- `results/satlas_sr_revalidation/statistical_analysis.json`
- `results/satlas_sr_revalidation/metric_correlations.csv`
- `results/satlas_sr_revalidation/visualizations_v2/*.png` (10 contact sheets)

---

## 8. Locating Master Reports & Data Tables

| Deliverable | File Path |
| :--- | :--- |
| **Baseline Freeze Manifest** | `results/satlas_sr_baseline_freeze_v1/baseline_manifest.json` |
| **Baseline Freeze Report** | `results/satlas_sr_baseline_freeze_v1/BASELINE_FREEZE_REPORT.md` |
| **Experiment Template** | `results/satlas_sr_baseline_freeze_v1/EXPERIMENT_TEMPLATE.md` |
| **Final Consistency Reconciliation** | `results/satlas_sr_reproduction_audit/final_consistency_reconciliation.md` |
| **Master Reproduction Report (v2)** | `reports/SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md` |
| **Paper Comparability Report** | `reports/satlas_sr_paper_comparability.md` |
| **Metric Deep-Dive Report** | `reports/satlas_sr_metric_validation_deep_dive.md` |
| **Per-Tile Metrics CSV (256 tiles)** | `results/satlas_sr_revalidation/per_tile_metrics.csv` |
| **Inference Manifest (256 tiles)** | `results/satlas_sr_full_validation/inference_manifest.csv` |
| **Spatial Registration CSV** | `results/satlas_sr_spatial_audit/registration_validation.csv` |
