# Satlas-SR Baseline Reproducibility Runbook (EXP-006)

This runbook documents the exact, verified procedure for setting up the Windows environment and reproducing the local baseline inference and evaluation pipeline on the Satlas-SR project.

> [!CAUTION]
> **Safety and Preservation Warning**:
> Never execute commands that overwrite existing baseline or experiment artifacts (`results/satlas_sr_baseline_freeze_v1/`, `results/satlas_sr_paper_metrics/`, `results/satlas_sr_revalidation/`, `results/satlas_sr_full_validation/`).
> Always direct new test outputs to a distinct directory (e.g., `results/satlas_sr_test_run/`).

---

## 1. Environment Setup (Windows PowerShell)

### Prerequisites
- OS: Windows 10/11 64-bit
- GPU: NVIDIA GPU with CUDA 12.1+ support (e.g., NVIDIA RTX 2000 Ada Generation)
- Dedicated Virtual Environment: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv`
- Repository Path: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`

### Step 1: Open PowerShell and Navigate to the Repository
```powershell
cd C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution
```

### Step 2: Activate the Virtual Environment
```powershell
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\Activate.ps1"
```
*Note: If PowerShell displays an ExecutionPolicy warning, you may execute python scripts directly using the virtual environment's Python binary:*
```powershell
$PYTHON = "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe"
```

### Step 3: Verify Environment and Hardware
```powershell
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```
**Expected Output:**
```text
PyTorch: 2.1.0+cu121, CUDA Available: True, Device: NVIDIA RTX 2000 Ada Generation
```

---

## 2. Model & Dataset Verification Commands

Before running any pipeline, confirm that the model weights and validation tiles exist and match expected checksums.

```powershell
# Verify Checkpoint SHA-256
certutil -hashfile experiments\esrgan_8S2.pth SHA256
```
**Expected Hash:**
`4099b244799c9ef89679fe26a71cb0a8ce885bb3c3dbff061bc4a7ffaa2d59ac`

```powershell
# Verify Dataset Count (256 tiles, 512 total files: 256 inputs, 256 targets)
(Get-ChildItem -Path "data\small_val_set" -File).Count
```
**Expected Output:** `512`

---

## 3. Verified Inference Command

The official inference entry point uses `ssr.eval.full_eval` (or `ssr.eval.evaluate_pipeline` via YAML configuration).

### Safe Inference Invocation (Directing to a New Directory)
To execute inference without overwriting existing baseline predictions:
```powershell
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" -m ssr.eval.full_eval `
    --opt options/test/test_esrgan_8S2.yml `
    --save_dir results/satlas_sr_test_run/predictions
```

### What This Command Does:
1. Loads checkpoint `experiments/esrgan_8S2.pth` (16,710,083 parameters, RRDBNet).
2. Reads 256 72-channel input tensors from `data/small_val_set/` (9 spectral bands $\times$ 8 timesteps at $128 \times 128$).
3. Performs $4\times$ super-resolution inference on GPU.
4. Generates 256 3-band RGB output images at $512 \times 512$ resolution.

---

## 4. Verified Metrics Evaluation Command

To evaluate cPSNR, standard PSNR, SSIM, VGG LPIPS, and CLIPScore on predicted outputs against ground truth targets:

### Safe Evaluation Invocation
```powershell
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" -m ssr.metrics.compute_paper_metrics `
    --pred_dir results/satlas_sr_test_run/predictions `
    --gt_dir data/small_val_set `
    --save_csv results/satlas_sr_test_run/per_tile_metrics.csv `
    --save_summary results/satlas_sr_test_run/metrics_summary.json `
    --crop_border 4 `
    --compute_clip `
    --compute_lpips
```

---

## 5. Validation Checks and Integrity Verification

After evaluation completes, run the following verification checks:

### Output Count Verification
```powershell
# Verify that all 256 predictions were generated
(Get-ChildItem -Path "results/satlas_sr_test_run/predictions" -Filter "*.png").Count
```
**Expected Output:** `256`

### Summary Metrics Inspection
```powershell
Get-Content "results/satlas_sr_test_run/metrics_summary.json" | ConvertFrom-Json | Select-Object -ExpandProperty summary
```
**Expected Target Baseline Values:**
- **cPSNR (crop 4)**: `20.2144 dB` ($\pm 4.1478$ dB)
- **Raw PSNR**: `18.8122 dB` ($\pm 3.8138$ dB)
- **Crop 4 PSNR**: `18.8870 dB` ($\pm 3.8600$ dB)
- **Raw SSIM**: `0.3950` ($\pm 0.1495$)
- **Crop 4 SSIM**: `0.3912` ($\pm 0.1564$)
- **VGG LPIPS**: `0.3470` ($\pm 0.0809$)
- **CLIPScore**: `0.9612` ($\pm 0.0344$)

---

## 6. Output Preservation Rules

1. **Frozen Artifact Protection**: Under no circumstances should any script or command modify files inside `results/satlas_sr_baseline_freeze_v1/`.
2. **Deterministic Comparisons**: All comparative investigations (such as dense vs. stride-2 search in EXP-001) must read from existing predictions and output to separate, numbered experiment folders (`results/satlas_sr_experiment_NNN/`).
3. **Artifact Integrity Logging**: Before and after executing experimental pipelines, generate an integrity log comparing file hashes and tile counts against `results/satlas_sr_experiment_006_final_audit/artifact_integrity_check.csv`.
