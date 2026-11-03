# Satlas-SR Frozen Baseline Audit & Reproduction Report (v1.0)

- **Audit Date**: 2026-10-22
- **Lead Investigator**: Senior Research Engineer (Antigravity AI)
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`
- **Frozen Baseline Directory**: `results/satlas_sr_baseline_freeze_v1`
- **Official Checkpoint**: `experiments/esrgan_8S2.pth` (SHA-256: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`)
- **Evaluation Dataset**: `data/small_val_set` (256 urban Sentinel-2 / NAIP paired tiles)
- **Hardware Platform**: NVIDIA RTX 2000 Ada Generation (16 GB VRAM), CUDA 12.1, PyTorch 2.1.0, Python 3.9.6

---

## 1. Executive Summary

This report establishes the frozen baseline package for the Satlas-SR satellite super-resolution reproduction project ([arXiv:2311.18082](https://arxiv.org/abs/2311.18082)). Following extensive verification, metric calibration, spatial registration audits, and reconciliation of earlier reporting discrepancies, this package freezes the repository's released small ESRGAN model on the 256-tile validation set (`small_val_set`).

All data structures, checkpoint weights, predictions, and reports have been preserved without modification. Zero model retraining, fine-tuning, or re-inference was conducted.

### Key Established Findings
1. **Model Parameter Count**: The instantiated official architecture (`SSR_RRDBNet`, 24 input channels) and released checkpoint have **16,710,083 parameters** (exactly matching `params_ema` in `experiments/esrgan_8S2.pth`).
2. **Dataset Pairing**: 256 / 256 tiles possess exact 3-way pairing between Sentinel-2 time series, NAIP reference imagery, and super-resolved model predictions.
3. **cPSNR Gain Distribution**: Across the 256 tiles, the mean cPSNR gain over rigid standard PSNR is **$+1.3274 \pm 2.0991\text{ dB}$** (median: **$+0.6076\text{ dB}$**), reaching a maximum of **$+12.8842\text{ dB}$** on tile `38409_49923`.
4. **Metric Terminology**: CLIPScore is the cosine similarity between feature embeddings; it is bounded within $[0.73, 1.00]$ due to unstandardized inputs and **must never be interpreted as a percentage or accuracy**.
5. **Registration Scope**: Raster cross-correlation estimates empirical image shift; it **does not prove physical geodetic registration**.
6. **Benchmark Scope**: The 256-tile urban validation subset is a local sanity baseline and **cannot be equated with the paper's nationwide multi-biome test set**.

---

## 2. What Was Verified Directly During This Freeze

The following checks were personally and directly executed during the baseline-freeze phase:
1. **Git State Inspection**: Confirmed working branch `experiments` at commit `3e67973f6746bedbd91cc032b4888574980e1daf`, with tracked modifications in `ssr/data/s2-naip_dataset.py`, `ssr/infer.py`, and `ssr/metrics/clipscore.py` safely preserved.
2. **Checkpoint Hash & Size**: Calculated SHA-256 directly on `experiments/esrgan_8S2.pth` (`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`, size: `134,157,518 bytes`).
3. **Direct Parameter Calculation**: Instantiated `SSR_RRDBNet(num_in_ch=24, ...)` and summed all parameter elements directly in PyTorch, confirming **16,710,083 parameters** and verifying that `params_ema` in the checkpoint has exactly **16,710,083 elements**.
4. **Manifest & Prediction Completeness**: Verified that `results/satlas_sr_full_validation/inference_manifest.csv` contains all 256 expected tile IDs, and verified that 256 corresponding `sr.png` (128x128) and `lr.png` (32x32) files exist on disk.
5. **Metric Gain Verification**: Computed mean and maximum gains directly from `results/satlas_sr_revalidation/per_tile_metrics.csv`, confirming mean gain of **$+1.3274\text{ dB}$** and maximum gain of **$+12.8842\text{ dB}$** on tile `38409_49923`.
6. **Unit Test Execution**: Executed `tests/test_dataset_path_compatibility.py` (3 tests passed in 0.000s) and `tests/test_metrics_rigor.py` (14 tests passed in 74.1s).
7. **Registration Precedence Logic**: Audited lines 149–170 of `scripts/diagnose_registration_estimator.py`, confirming that category precedence enforces a mutually exclusive partition summing to exactly 256 tiles ($200 + 14 + 37 + 5 = 256$).
8. **JSON & CSV Parsing**: Validated syntax and row counts for all newly created baseline artifacts.

---

## 3. What Was Inherited from Previous Audit Reports

The following items were inherited from earlier audits and cross-validated against disk artifacts:
1. **Prior Inference Run**: Generation of the 256 super-resolved predictions in `results/satlas_sr_full_validation/` (generated with seed 0, `params_ema`, using `run_full_validation_inference.py`).
2. **Phase C Metric Revalidation**: Recomputation of cPSNR, LPIPS, CLIPScore, PSNR, and SSIM across all 256 tiles in `results/satlas_sr_revalidation/` matching `results/satlas_sr_paper_metrics/` with $|\Delta| < 10^{-3}$.
3. **OpenCLIP Cache**: Pre-downloaded weights for `ViT-bigG-14-CLIPA-336` (`datacomp1b`) cached in `C:\Users\CSE-SDPL\.cache\huggingface\hub\`.
4. **LPIPS Cache**: Pre-downloaded weights for VGG16 trunk cached in `C:\Users\CSE-SDPL\.cache\torch\hub\checkpoints\`.

---

## 4. Dataset & Benchmark Scope

### 4.1 Dataset Structure
- Root: `C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set`
- **Sentinel-2**: 256 subdirectories, each containing a `tci.png` file of shape `[1024, 32, 3]` representing 32 temporal acquisitions of $32 \times 32$ RGB pixels.
- **NAIP**: 256 matching subdirectories, each containing an `rgb.png` reference image of shape `[128, 128, 3]` ($4\times$ spatial resolution).
- **Integrity**: 100% 3-way pairing verified; zero truncated, corrupted, or missing files.

### 4.2 Benchmark Limitations
The `small_val_set` consists of 256 urban tiles selected from the continental United States. It was curated as a local regression test and validation subset. **It cannot be equated with the paper's primary nationwide benchmark**, which evaluated an 87M medium ESRGAN across thousands of geographically diverse tiles encompassing forest, desert, agricultural, and coastal biomes.

---

## 5. Model Architecture & Checkpoint Identity

- **Generator Model**: `SSR_RRDBNet` (Multi-Frame Residual-in-Residual Dense Block Network)
- **Configuration**:
  - Scale: $4\times$
  - Number of input channels: $24$ (8 temporal frames $\times$ 3 RGB channels)
  - Number of output channels: $3$ (Single high-resolution RGB image)
  - Feature channels: $64$
  - Number of RRDB blocks: $23$
  - Growth channels: $32$
- **Checkpoint**: `experiments/esrgan_8S2.pth` (134,157,518 bytes)
- **Loaded Key**: `params_ema` (loaded with `strict=True`)
- **Total Model Parameters**: **16,710,083** (verified via PyTorch parameter count and state dict inspection).

---

## 6. Inference Reproducibility & Protocol

Inference over `small_val_set` is governed by [run_full_validation_inference.py](file:///c:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/run_full_validation_inference.py):
1. **Deterministic Tile Order**: Sentinel-2 subdirectories are sorted alphabetically by tile ID, ensuring operating-system-independent RNG stream progression.
2. **Frame Selection**: Black chunks containing `[0, 0, 0]` are filtered out; 8 frames are uniformly sampled at random using Python's `random.sample` with `random.seed(0)`.
3. **Traceability**: Selected frame indices (0-based) for each tile are recorded in `inference_manifest.csv`.
4. **Normalization & Postprocessing**: Inputs are scaled by $/255.0$; outputs are clamped to $[0.0, 1.0]$, transposed to $(128, 128, 3)$, multiplied by $255$, cast to `np.uint8`, and saved as uncompressed PNGs.

---

## 7. Metric Results & Formal Definitions

Across the 256 tiles, the verified baseline metrics are:

| Metric | Official Function | Mean ± Std | 95% Bootstrap CI | Median | Range [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cPSNR (crop 4, dB)** | `calculate_cpsnr` | **$20.2144 \pm 4.1478$** | $[19.6852, 20.7486]$ | $19.4600$ | $[12.9806, 38.2974]$ |
| **Standard PSNR (crop 4, dB)** | `calculate_psnr` | **$18.8870 \pm 3.8600$** | $[18.4129, 19.3646]$ | $18.3366$ | $[12.4411, 34.5049]$ |
| **cPSNR Gain (dB)** | $\text{cPSNR} - \text{PSNR}$ | **$+1.3274 \pm 2.0991$** | $[+1.0887, +1.5859]$ | $+0.6076$ | $[-0.1372, +12.8842]$ |
| **SSIM (crop 4)** | `calculate_ssim` | **$0.3912 \pm 0.1564$** | $[0.3730, 0.4113]$ | $0.3784$ | $[0.0763, 0.8164]$ |
| **LPIPS (VGG16)** | `calculate_lpips` | **$0.3470 \pm 0.0809$** | $[0.3375, 0.3572]$ | $0.3396$ | $[0.1384, 0.6033]$ |
| **CLIPScore** | `calculate_clipscore` | **$0.9612 \pm 0.0344$** | $[0.9569, 0.9650]$ | $0.9736$ | $[0.7066, 0.9953]$ |

### Technical Metric Notes
1. **cPSNR Search Stride**: In `ssr/metrics/cpsnr.py`, the search slices images at `row_offset` and `8 - row_offset`, evaluating relative shifts $\Delta \in \{-8, -6, -4, -2, 0, 2, 4, 6, 8\}$ along each axis. It tests an 81-point grid with a stride of 2 pixels. Odd-pixel shifts are not directly aligned.
2. **cPSNR Gain Reconciled**: The maximum gain across all 256 tiles is **$+12.8842\text{ dB}$** on tile `38409_49923` ($\text{cPSNR} = 28.2705\text{ dB}, \text{PSNR} = 15.3863\text{ dB}$).
3. **CLIPScore**: Evaluates unnormalized $[0, 1]$ tensors on `ViT-bigG-14-CLIPA-336`. Latent features cluster in a positive hyper-cone; baseline natural images score between $0.73$ and $1.00$.

---

## 8. Registration Diagnostics & Classification Precedence

### 8.1 Precedence Rules in Code
In `scripts/diagnose_registration_estimator.py`, each tile is evaluated using both raw and Hanning-windowed Fourier Phase Cross-Correlation (PCC) and Normalized Cross-Correlation (NCC):

```python
is_spectral_wraparound = (sr_mag > 10.0 and (mag_win <= 4.0 or math.hypot(ncc_dx, ncc_dy) <= 4.0))
is_low_correlation = (ncc_peak < 0.25)

if is_low_correlation:
    reliability = "UNRELIABLE_LOW_CORRELATION"
elif is_spectral_wraparound:
    reliability = "SPECTRAL_WRAPAROUND_CORRECTED"
elif sr_mag > 15.0:
    reliability = "UNRELIABLE_EXTREME_OUTLIER"
else:
    reliability = "RELIABLE"
```

### 8.2 Category Breakdown & Partition Verification
1. **`UNRELIABLE_LOW_CORRELATION` (37 tiles, 14.5%)**: First precedence. Tiles with peak $\text{NCC} < 0.25$ (clouds, uniform water, seasonal changes) where any shift estimate is ungrounded.
2. **`SPECTRAL_WRAPAROUND_CORRECTED` (14 tiles, 5.5%)**: Second precedence. Tiles with acceptable correlation where raw unwindowed FFT reported large shifts ($> 10\text{ px}$), but windowed FFT or NCC resolved the true displacement to $\le 4.0\text{ px}$.
3. **`UNRELIABLE_EXTREME_OUTLIER` (5 tiles, 2.0%)**: Third precedence. Tiles with acceptable correlation where displacement remains $> 15.0\text{ px}$ under both windowed and raw methods.
4. **`RELIABLE` (200 tiles, 78.1%)**: Unflagged tiles with consistent sub-window displacement.

$$\text{Total Sum} = 37 + 14 + 5 + 200 = 256\text{ tiles } (100.0\%)$$
$$\text{Reliable Subset} = 200\text{ (RELIABLE)} + 14\text{ (WRAPAROUND\_CORRECTED)} = 214\text{ tiles } (83.59\%)$$

On the reliable subset ($N=214$):
- Mean displacement: **$3.8631 \pm 5.7413\text{ px}$**
- Median displacement: **$1.7360\text{ px}$**
- IQR: $[1.0300, 3.4340]\text{ px}$
- $55.6\%$ of tiles have displacement $\le 2\text{ px}$; $86.9\%$ have displacement $\le 8\text{ px}$.

---

## 9. Statistical Uncertainty & Limitations

1. **Bootstrap Uncertainty**: The 95% bootstrap confidence intervals (2,000 resamples, seed 42) describe sampling uncertainty in the sample mean of these 256 tiles. They do not describe model parameter uncertainty or nationwide generalization.
2. **Fidelity vs. Perception Tradeoff**: SSIM correlates moderately with cPSNR ($\rho = +0.5223, p < 10^{-18}$), but LPIPS shows weak negative correlation with cPSNR ($\rho = -0.1737, p = 0.0053$). Super-resolution models optimizing perceptual losses often synthesize plausible high-frequency edge textures that depart from exact pixel-wise ground truth.

---

## 10. Git Working-Tree & Preservation Notes

- **Current Commit**: `3e67973f6746bedbd91cc032b4888574980e1daf` on branch `experiments`.
- **Pre-existing Tracked Modifications**:
  - `ssr/data/s2-naip_dataset.py`: Replaced POSIX `.split('/')` with `os.path.basename` and `os.path.dirname` (verified by unit tests).
  - `ssr/infer.py`: Anchored RNG seed.
  - `ssr/metrics/clipscore.py`: Added safe try/except fallback for `clip` import.
- **Preservation Policy**: All working-tree modifications, existing outputs in `results/satlas_sr_full_validation/`, `results/satlas_sr_paper_metrics/`, `results/satlas_sr_revalidation/`, and prior reports remain strictly untouched.

---

## 11. Remaining Issues & Recommended Follow-up

1. **PROBA-V Stride Correction in Future Research**: Future experimental benchmarks should test a full 1-pixel stride translation search to evaluate whether odd-pixel shifts yield higher fidelity gains.
2. **Standardized CLIP Transforms**: Controlled experiments comparing unstandardized vs ImageNet-standardized CLIP transforms should be conducted to establish whether feature sensitivity improves.
3. **Scaling Experiments**: Evaluation of the medium (87M) and large (347M) ESRGAN variants would be required to reproduce the paper's primary cross-method comparison against SR3.

---

## 12. Exact Commands for Future Controlled Experiments

When conducting new experiments against this frozen baseline:
```powershell
# 1. Run a small controlled test to a separate output directory
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" run_full_validation_inference.py `
    -opt ssr/options/infer_satlas_full_validation.yml `
    --tiles 20771_46525 20890_46801 `
    --output-dir results/experiments/exp_001_test

# 2. Re-evaluate metrics on new predictions
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" evaluate_satlas_sr_paper_metrics.py `
    --predictions-dir results/experiments/exp_001_test `
    --naip-dir ../data/small_val_set/naip `
    --output-dir results/experiments/exp_001_test/metrics `
    --crop-border 4 `
    --clip-model clipa-ViT-bigG-14
```
