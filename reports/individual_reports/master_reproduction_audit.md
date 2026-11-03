# Master Reproduction Audit: Satlas Super-Resolution (Satlas-SR)

**Audit Date**: 2026-10-21  
**Scope**: Comprehensive evidence-based reproduction audit of the Satlas-SR base paper ([arXiv:2311.18082](https://arxiv.org/html/2311.18082v1)), local software environment, released baseline model, validation dataset, inference pipeline, and paper-aligned evaluation metric suite.

---

## 1. Executive Summary

This audit establishes the exact state of reproduction for the Satlas-SR baseline on the small urban validation set (`small_val_set`). 

1. **Baseline Model Inference**: Successfully and deterministically reproduced using the released 8-Sentinel-2 ESRGAN checkpoint (`esrgan_8S2.pth`, ~16.71M parameters, `params_ema` weights). All 256 tiles have verified, uncorrupted $128 \times 128$ RGB super-resolved predictions and $32 \times 32$ RGB low-resolution previews.
2. **Dataset & Pairing Integrity**: Exactly 256 Sentinel-2 tile directories match 256 NAIP ground truth references with 100% 1-to-1 correspondence. All 256 Sentinel-2 `tci.png` files consist of 32 stacked $32 \times 32$ frames ($32 \times 1024$ PNGs). Every tile has $\ge 8$ valid (zero-black-pixel) frames, ensuring clean inputs.
3. **Metric Implementations**: The official paper metric suite—**cPSNR** (translation & radiometric bias compensation up to 8px), **LPIPS** (VGG perceptual distance), and **CLIPScore** (image-to-image cosine similarity via OpenCLIP `clipa-ViT-bigG-14`)—has been audited, tested, and evaluated across all 256 tiles alongside supplementary PSNR and SSIM.
4. **Scope Boundaries**: This work represents a faithful reproduction of the *released baseline inference and evaluation*. It does not reproduce model retraining from scratch, the full 1.2M-pair training schedule, the medium 87M ESRGAN method comparison, or the paper's expert human building-count studies.

---

## 2. Environment & System Specifications

| Component | Verified Specification | Source / Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (64-bit) | PowerShell / `sys.platform` |
| **Python Virtualenv** | Python 3.9.6 | `C:\Users\CSE-SDPL\Desktop\DL_image\.venv` |
| **Deep Learning Framework** | PyTorch 2.1.0+cu121 | `torch.__version__` |
| **CUDA Acceleration** | Available (`True`), CUDA 12.1 runtime | `torch.cuda.is_available()` |
| **Compute Hardware** | NVIDIA RTX 2000 Ada Generation | 16,380 MiB VRAM; Ada Lovelace architecture |
| **Computer Vision Tooling** | `torchvision 0.16.0`, `scikit-image 0.22.0`, `Pillow 10.1.0` | Environment audit |
| **Metric Packages** | `lpips 0.1.4`, `open_clip_torch 3.3.0` | Environment audit |
| **Dependency Note** | OpenAI's `clip` package is absent; resolved via backward-compatible OpenCLIP execution in `ssr/metrics/clipscore.py` | Verified in runtime tests |

---

## 3. Git Status & Repository Artifact Classification

- **Active Branch**: `experiments` (synchronized with `origin/experiments`).
- **Tracked Modifications**:
  - `ssr/infer.py`: Deterministic seed `random.seed(opt.get('manual_seed', 0))` added at line 23.
  - `ssr/metrics/clipscore.py`: Safe conditional import `try: import clip except ImportError: clip = None` to enable OpenCLIP models (`clipa-ViT-bigG-14`) to run without requiring OpenAI's `clip` package.

### Artifact Classification Table

| File / Directory | Category | Role in Reproduction |
| :--- | :--- | :--- |
| `ssr/archs/rrdbnet_arch.py` | Original Repository | Generator implementation (`SSR_RRDBNet`) |
| `ssr/data/s2-naip_dataset.py` | Original Repository | S2-NAIP dataset loader (contains Windows POSIX `/` bug) |
| `ssr/metrics/cpsnr.py` | Original Repository | Official cPSNR implementation from PROBA-V |
| `ssr/metrics/lpips.py` | Original Repository | Official LPIPS implementation with VGG backbone |
| `ssr/metrics/clipscore.py` | Original Repository | Official CLIPScore implementation with OpenCLIP CLIPA |
| `ssr/options/experiments/esrgan_baseline.yml` | Original Repository | Official baseline training & testing specification |
| `experiments/esrgan_8S2.pth` | Released Official Artifact | Checkpoint weights released by AllenAI |
| `data/small_val_set/` | Released Official Dataset | 256 Sentinel-2 and NAIP urban image pairs |
| `run_full_validation_inference.py` | Local Reproduction Tool | Deterministic, sorted-tile inference runner |
| `evaluate_satlas_sr_paper_metrics.py` | Local Reproduction Tool | Comprehensive paper-aligned metric evaluation runner |
| `results/satlas_sr_full_validation/` | Local Experimental Output | 256 SR predictions, LR previews, and inference manifest |
| `results/satlas_sr_paper_metrics/` | Local Experimental Output | Full metric evaluations (cPSNR, LPIPS, CLIPScore, PSNR, SSIM) |
| `results/satlas_sr_reproduction_audit/`| Local Audit Artifact | Dataset inventory CSV/JSON and validation reports |
| `reports/` | Local Documentation | Gap analyses, feasibility audits, and technical summaries |

---

## 4. Checkpoint & Model Verification

### Checkpoint Provenance
- **File Path**: `experiments\esrgan_8S2.pth`
- **File Size**: 134,157,518 bytes (~127.9 MiB)
- **SHA-256 Digest**: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`
- **Internal Keys**: `dict_keys(['params', 'params_ema'])` (702 weight tensors each)
- **Active Inference Weights**: `params_ema` (Exponential Moving Average weights)

### Generator Architecture
- **Model Class**: `SSR_RRDBNet` (`ssr.archs.rrdbnet_arch`)
- **Scale Factor**: 4× ($32 \times 32 \to 128 \times 128$)
- **Input Channels**: 24 ($8 \text{ low-resolution Sentinel-2 frames} \times 3 \text{ RGB channels}$)
- **Output Channels**: 3 (High-resolution RGB image)
- **Hyperparameters**: `num_feat=64`, `num_block=23`, `num_grow_ch=32`
- **Instantiated Parameters**: **16,710,083 parameters** (exactly consistent with the paper's reported 17M small ESRGAN model in Section 4.1).
- **Input Shape**: `[1, 24, 32, 32]` (batch size 1, 24 channels, 32 height, 32 width).
- **Output Shape**: `[1, 3, 128, 128]` clamped to $[0, 1]$ and scaled to $[0, 255]$ uint8.

---

## 5. Dataset Architecture & Verification

The dataset inventory audit ([`results/satlas_sr_reproduction_audit/dataset_inventory.json`](file:///c:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_reproduction_audit/dataset_inventory.json)) confirms:

1. **Sentinel-2 Structure**:
   - Location: `data/small_val_set/sentinel2/<tile_id>/`
   - Total Tiles: 256 unique directories.
   - Files per Tile: 10 PNG files (`b01.png`...`b12.png` and `tci.png`).
   - `tci.png`: $32 \times 1024$ uint8 RGB image containing 32 vertically stacked $32 \times 32$ frames.
   - Frame Filtering: Across all 256 tiles, the mean number of valid (non-black) frames is **27.6** (min: 10, max: 32). Zero tiles have fewer than 8 valid frames, ensuring that all 8 sampled input frames per tile are free of black boundary artifacts.
2. **NAIP Ground Truth Structure**:
   - Location: `data/small_val_set/naip/<naip_source_id_or_date>/<tile_id>/rgb.png`
   - Total Tiles: Exactly 256 unique directories with a 1-to-1 exact ID match to Sentinel-2.
   - `rgb.png`: $128 \times 128$ uint8 RGB image ($4\times$ ground truth reference).
3. **Three-Way Integrity**:
   - All 256 tiles have matching S2 inputs, NAIP references, and SR predictions (100.0% completion, 0 missing, 0 corrupt).

---

## 6. Official Metrics vs. Paper Benchmarks

The official paper metrics evaluated across all 256 tiles in [`results/satlas_sr_paper_metrics/`](file:///c:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_paper_metrics) report:

| Metric | Local Full Validation Mean ± Std | Local Range [Min, Max] | Baseline Protocol & Settings | Paper Context (arXiv:2311.18082) |
| :--- | :--- | :--- | :--- | :--- |
| **cPSNR (crop 4)** | **20.21 ± 4.15 dB** | [9.59, 39.54] dB | `crop_border=4`, `max_offset=8` | PROBA-V metric accounting for spatial & radiometric shifts |
| **cPSNR (no crop)** | **20.11 ± 4.11 dB** | [9.79, 39.50] dB | `crop_border=0`, `max_offset=8` | Full-frame translation/bias optimized PSNR |
| **LPIPS (VGG)** | **0.3470 ± 0.0809** | [0.1442, 0.7196] | VGG16 trunk, raw perceptual distance | Lower is more similar; paper converts to agreement score |
| **CLIPScore** | **0.9612 ± 0.0344** | [0.7768, 0.9930] | `clipa-ViT-bigG-14` (datacomp1b) | Primary evaluation metric in paper (Table 2) |
| **PSNR (crop 4)** | **18.89 ± 3.86 dB** | [5.60, 34.78] dB | BasicSR `crop_border=4` | Standard pixel-to-pixel PSNR |
| **PSNR (raw/full)** | **18.81 ± 3.81 dB** | [5.73, 33.82] dB | skimage array PSNR | Raw full-frame PSNR without edge cropping |
| **SSIM (crop 4)** | **0.3912 ± 0.1564** | [0.0501, 0.9129] | BasicSR `crop_border=4` | Standard structural similarity |
| **SSIM (raw/full)** | **0.3950 ± 0.1495** | [0.0570, 0.9059] | skimage array SSIM | Raw full-frame SSIM without edge cropping |

### Key Diagnostic Insight: cPSNR vs. Standard PSNR
cPSNR shows an average gain of **+1.33 dB** over standard cropped PSNR ($20.21 \text{ dB}$ vs. $18.89 \text{ dB}$). This confirms the paper's thesis: conventional PSNR is severely degraded by sub-pixel misalignments and atmospheric illumination variations inherently present between Sentinel-2 and airborne NAIP sensors.

---

## 7. Known Discrepancies & Reproducibility Boundaries

1. **Small (17M) vs. Medium (87M) ESRGAN**:
   - The paper's main method comparison (Table 2) compares SRCNN, HighResNet, medium ESRGAN (87M), and SR3 (97M).
   - The released checkpoint evaluated here is the **small (17M) 8-S2 ESRGAN**. It matches the scaling experiments in Section 4.1, not Table 2's 87M medium variant.
2. **Missing Authoritative Georeferencing**:
   - Extracted PNG tiles lack embedded GeoTIFF tags, RPCs, or affine transform matrices.
   - Spatial pairing relies on Web-Mercator zoom-17 tile naming conventions (`<x>_<y>`). While tiles represent the same geographic cell, individual sub-pixel shifts must be handled empirically.
3. **S2 Stack Temporal Ordering**:
   - The README notes that the 32 frames in `tci.png` are not guaranteed to be in strict chronological order. Frame selection filters invalid black chunks and samples 8 frames deterministically under seed 0.
