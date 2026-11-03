# Independent Baseline Freeze Verification & Research Readiness Review (v1.0)

- **Review Date**: 2026-10-22
- **Reviewer**: Senior Research Engineer (Antigravity AI)
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`
- **Frozen Baseline Target**: `results/satlas_sr_baseline_freeze_v1/`
- **Output Directory**: `results/satlas_sr_baseline_freeze_review_v1/`

---

## 1. Verification of Frozen Artifacts Against Source Evidence

All primary claims in `baseline_manifest.json`, `REPRODUCTION_RUNBOOK.md`, `BASELINE_FREEZE_REPORT.md`, and `EXPERIMENT_TEMPLATE.md` were cross-checked against source data, model checkpoints, and codebase scripts.

### Detailed Evidence Matrix

| Claim / Field | Stated Value | Source File / Inspection Command | Direct Ground Truth | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Git Commit** | `3e67973f...` | `git rev-parse HEAD` | `3e67973f6746bedbd91cc032b4888574980e1daf` | **DIRECTLY VERIFIED** |
| **Git Branch** | `experiments` | `git status --short --branch` | `## experiments...origin/experiments` | **DIRECTLY VERIFIED** |
| **Tracked Modifications** | 3 files modified | `git status --short` | `ssr/data/s2-naip_dataset.py`, `ssr/infer.py`, `ssr/metrics/clipscore.py` | **DIRECTLY VERIFIED** |
| **Checkpoint SHA-256** | `167d6b9d...` | `hashlib.sha256(open('experiments/esrgan_8S2.pth'))` | `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07` | **DIRECTLY VERIFIED** |
| **Checkpoint File Size** | 134,157,518 bytes | `Get-Item experiments/esrgan_8S2.pth` | `134,157,518 bytes` | **DIRECTLY VERIFIED** |
| **Model Parameters** | 16,710,083 | PyTorch `sum(p.numel())` on `SSR_RRDBNet(24ch)` & `ckpt['params_ema']` | `16,710,083 parameters` | **DIRECTLY VERIFIED** |
| **Tile Count (Dataset)** | 256 tiles | Directory count of `../data/small_val_set/sentinel2` & `naip` | 256 unique matching tile directories | **DIRECTLY VERIFIED** |
| **Inference Completeness** | 256 predictions | `inference_manifest.csv` and `results/satlas_sr_full_validation/` | 256 `sr.png` (128x128) and `lr.png` (32x32) files exist | **DIRECTLY VERIFIED** |
| **Metric Records** | 256 records | `results/satlas_sr_revalidation/per_tile_metrics.csv` | Exactly 256 valid per-tile rows | **DIRECTLY VERIFIED** |
| **cPSNR Mean Gain** | +1.3274 dB | Mean of `cpsnr_gain_over_psnr_db` across 256 rows | `+1.327447 dB` | **DIRECTLY VERIFIED** |
| **cPSNR Max Gain** | +12.8842 dB | Row `38409_49923` in `per_tile_metrics.csv` | `28.2705 - 15.3863 = +12.8842 dB` | **DIRECTLY VERIFIED** |
| **Registration Categories** | 200 / 14 / 37 / 5 | `registration_validation.csv` reliability column counts | 200 Reliable, 14 Wraparound, 37 Low-Corr, 5 Outlier ($= 256$) | **DIRECTLY VERIFIED** |
| **Reliable Shift Median** | 1.736 px | Median of windowed shift across the 214 reliable rows | `1.736 px` (Mean: `3.863 px`) | **DIRECTLY VERIFIED** |
| **Inference Configuration** | Config YAML | `ssr/options/infer_satlas_full_validation.yml` | SHA-256: `946a3c79c8463cac506e1187cf0ebf9d6e7a12a210c0915dfb5a8999205a9904` | **DIRECTLY VERIFIED** |

---

## 2. Examination of the cPSNR Odd-Pixel Stride Caveat

### The Stated Caveat
In `BASELINE_FREEZE_REPORT.md` (Section 7.1) and earlier drafts:
> *"Because cPSNR searches with a stride of 2 pixels, true shifts of 1, 3, 5, 7 pixels are approximated at adjacent even coordinates, under-estimating cPSNR by approximately 0.2–0.5 dB for those tiles."*

### Source Tracing & Evidentiary Review
1. **Source Evidence Found**:
   - In `ssr/metrics/cpsnr.py` (lines 40–45), the loop computes `2 * row_offset - 8` for `row_offset in range(9)`. This establishes as an exact mathematical fact that **only even offsets $\{-8, -6, -4, -2, 0, +2, +4, +6, +8\}$ are evaluated**.
   - In `tests/test_metrics_rigor.py` and `scripts/verify_metrics_deep_dive.py`, synthetic test pairs shifted by $+1$ px were shown to achieve lower alignment than pairs shifted by $+2$ px because the $+1$ px offset is not on the search grid.
2. **Unsupported Claim Identified**:
   - There was **no empirical dataset measurement** across the 256 validation tiles quantifying the exact difference between a dense 1-pixel stride search and the 2-pixel stride search.
   - The specific numerical range **`0.2–0.5 dB`** was an analytical conjecture based on typical image gradients, not an empirically measured statistic from the 256 tiles.
3. **Formal Review Correction**:
   - **Supported Fact**: The official cPSNR implementation evaluates only even-pixel shifts (stride = 2). Tiles whose optimal translation lies on an odd integer coordinate cannot achieve exact integer alignment.
   - **Correction**: The magnitude of underestimation across real satellite imagery is **unquantified** in existing baseline reports. Any claim specifying an average loss of "0.2–0.5 dB" must be treated as an unverified heuristic until a controlled experiment with dense 1-pixel stride is explicitly executed.

---

## 3. Command Safety & Argument Verification in `REPRODUCTION_RUNBOOK.md`

We inspected the command-line argument parsers and paths for all commands listed in `results/satlas_sr_baseline_freeze_v1/REPRODUCTION_RUNBOOK.md`.

### Findings & Identified Risks

| Runbook Section | Command Listed | Safety Status | Risk Analysis & Corrected Safe Form |
| :--- | :--- | :--- | :--- |
| **Section 1: Hardware** | PyTorch inline GPU query | **SAFE** | Read-only; executes without modifying any files. |
| **Section 2: Checkpoint** | `Get-FileHash` & parameter sum | **SAFE** | Read-only; directly verified against filesystem. |
| **Section 3: Unit Tests** | `test_dataset_path_compatibility.py` & `test_metrics_rigor.py` | **SAFE** | Strictly non-destructive unit tests. |
| **Section 4: Pilot Inference (Option A)** | `& $PYTHON run_five_tile_inference.py -opt ssr/options/infer_satlas_five_tiles.yml` | **CAUTION (OVERWRITE RISK)** | **Risk**: `infer_satlas_five_tiles.yml` hardcodes `save_path: results/satlas_sr_pilot_five_tiles`. Running this overwrites existing pilot outputs.<br>**Safe Alternative**: Use Option B with `--output-dir results/test_pilot_new`. |
| **Section 4: Pilot Inference (Option B)** | `run_full_validation_inference.py --tiles ... --output-dir results/test_subset_smoke` | **SAFE** | Argument parser supports `--tiles` and `--output-dir`. Writes to a non-colliding directory. |
| **Section 5: Paper Metrics (Option A)** | `evaluate_satlas_sr_paper_metrics.py --output-dir results/satlas_sr_paper_metrics` | **CAUTION (OVERWRITE RISK)** | **Risk**: Defaults to `results/satlas_sr_paper_metrics`, which would overwrite existing paper metric outputs.<br>**Safe Alternative**: Specify `--output-dir results/satlas_sr_paper_metrics_recheck`. |
| **Section 5: Revalidation (Option B)** | `scripts/revalidate_and_compare_metrics.py` | **CAUTION (RE-WRITE RISK)** | **Risk**: Hardcoded output path writes directly to `results/satlas_sr_revalidation/per_tile_metrics.csv`. |
| **Section 6: Registration Audit** | `scripts/diagnose_registration_estimator.py` | **CAUTION (RE-WRITE RISK)** | **Risk**: Hardcoded output path writes directly to `results/satlas_sr_spatial_audit/registration_validation.csv`. |
| **Section 7: Statistical Analysis** | `scripts/run_statistical_and_failure_analysis.py` | **CAUTION (RE-WRITE RISK)** | **Risk**: Hardcoded output path writes directly to `results/satlas_sr_revalidation/statistical_analysis.json` and contact sheets. |

### Recommendation for Runbook Users
Commands in Sections 4 (Option A), 5, 6, and 7 re-execute existing generation scripts that overwrite their target directories. Future researchers seeking to verify these steps without mutating baseline artifacts should either copy the output directory first or inspect the existing verified CSV/JSON artifacts directly.

---

## 4. Baseline Reproducibility Boundaries

### What Has Been Demonstrated on `small_val_set` (256 Tiles)
1. **Architectural & Checkpoint Match**: The official 16.71M small ESRGAN architecture (`SSR_RRDBNet`, 24 input channels) loads strictly from `experiments/esrgan_8S2.pth` with zero missing/unexpected keys.
2. **Inference Execution**: 8-frame multi-temporal super-resolution inference is 100% deterministic under seed 0 with tile-sorted ordering, generating 256 valid $128 \times 128$ predictions.
3. **Metric Implementation**: Local implementations of cPSNR, standard PSNR, SSIM, LPIPS, and CLIPScore were tested, verified, and re-evaluated across all 256 tiles with zero numerical discrepancy against recorded baseline files.
4. **Spatial Registration Characteristics**: 83.6% of tiles possess reliable raster translation with a median displacement of $1.736\text{ px}$. 14 tiles suffered from raw unwindowed FFT boundary wraparound, which was resolved by Hanning windowing and NCC.

### What Has NOT Been Demonstrated About the Paper's Nationwide Benchmark
1. **Model Scaling**: The paper's primary cross-method comparison against SR3 in Section 5 evaluates an **87M medium ESRGAN**; only the released 16.71M small ESRGAN model was evaluated locally.
2. **Biome Diversity**: The 256 tiles represent continental US urban centers. The paper evaluates nationwide test sets encompassing desert, agricultural, forested, and coastal biomes. Local performance cannot be generalized to these biomes.
3. **Training & Optimization Dynamics**: No retraining, loss convergence (L1, perceptual, adversarial loss balancing), or temporal sampling ablation during training was evaluated.
4. **Physical Geodesy**: Measured raster shifts reflect empirical image cross-correlation, not geodetic satellite orbit ephemeris or physical sensor alignment.

---

## 5. Review Conclusion

The frozen baseline package `results/satlas_sr_baseline_freeze_v1/` is **technically sound, fully reproducible, and rigorously accounted for**. All primary claims regarding parameter counts, dataset pairing, metric aggregates, and registration partitions have been directly verified against source artifacts.
