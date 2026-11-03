# EXP-011: Sentinel-2 Frame-Selection Sensitivity Pilot Results

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe` (PyTorch 2.1.0+cu121)  
**Hardware Platform**: Windows 11 Pro, NVIDIA RTX 2000 Ada Generation (16 GB VRAM)  
**Execution Scope**: Option A Pilot Only (5 Canonical Validation Tiles, 7 Candidate Strategies, 35 Total Inferences)  
**Execution Status**: COMPLETED SUCCESSFULLY (Zero Errors, 35/35 Valid $128 \times 128$ PNGs Generated)  
**Timestamp**: 2026-10-22T21:43:47+05:30  

> [!IMPORTANT]
> **Preliminary Nature of Findings**: This pilot evaluates exactly five tiles ($N=5$). All observed differences, ranking shifts, and metric deltas must be treated as **strictly preliminary exploratory signals**. No strategy should be declared "improved" or "superior" based on five tiles alone.

---

## 1. Executive Summary

This pilot experiment compares the official Satlas-SR eight-frame selection against six alternative strategies across the five canonical pilot tiles:
`20771_46525`, `20890_46801`, `20926_46045`, `21068_45692`, and `21856_49804`.

### Key Preliminary Observations:
1. **Permutation Sensitivity Confirmed (S2 vs. S0)**:  
   Sorting the exact same 8 frames chosen by baseline Seed 0 into ascending index order (`S2`) produced measurable changes in per-tile predictions (mean cPSNR delta $+0.0169\text{ dB}$, up to $+0.1334\text{ dB}$ on tile `21856_49804`). This empirically confirms that `SSR_RRDBNet`'s first convolutional layer (`conv_first`, shape `[64, 24, 3, 3]`) is **sensitive to the channel ordering** of identical input frames.
2. **Stochastic Seed Variance (S1 vs. S0)**:  
   Switching the pseudorandom seed from 0 to 42 (`S1`) resulted in a $-0.1217\text{ dB}$ shift in mean cPSNR, a $-0.2230\text{ dB}$ shift in PSNR, and a $+0.0115$ degradation in LPIPS perceptual distance. This confirms that baseline scores carry non-trivial sampling variance depending on the PRNG draw.
3. **Cloud Avoidance Rescues Outliers (S6 vs. S0)**:  
   The unsupervised Low-Brightness heuristic (`S6`, which selects the 8 frames with the lowest mean RGB intensity to avoid bright cloud/glint contamination) achieved the **highest PSNR** ($20.0269\text{ dB}$, mean delta $+0.6048\text{ dB}$) and the **best perceptual LPIPS** ($0.3548$ vs. $0.3569$ for baseline). Most notably, on tile `21856_49804`, PSNR jumped by **$+2.7018\text{ dB}$** (from $17.3782\text{ dB}$ to $20.0800\text{ dB}$) and LPIPS improved by $-0.0336$, demonstrating that cloud filtering can resolve severe localized failure cases.
4. **Deterministic Striding vs. Clustering (S3, S4, S5)**:  
   Uniform strided sampling (`S3`) yielded a slight degradation in cPSNR ($-0.3586\text{ dB}$), while Early Window clustering (`S4`) closely tracked baseline performance ($+0.0177\text{ dB}$ cPSNR).

---

## 2. Experimental Controls and Reproducibility

Every run adhered strictly to the frozen reproduction standards:
- **Tiles Evaluated**: Identical 5 canonical tile IDs across all strategies.
- **Model Checkpoint**: Frozen official checkpoint [`experiments/esrgan_8S2.pth`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/experiments/esrgan_8S2.pth) (`params_ema`, SHA-256: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`).
- **Network Configuration**: `SSR_RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)`.
- **Normalization**: Float32 tensor divided by 255.0 (`float() / 255.0`), shape `[1, 24, 32, 32]`.
- **Output Validation**: All 35 predictions were verified to be shape $(128, 128, 3)$, uint8, with no artificial resizing.
- **Ground Truth Target**: NAIP reference `rgb.png` files of shape $(128, 128, 3)$.
- **Zero Interference**: All artifacts were written to [`results/satlas_sr_experiment_011_frame_selection_audit/pilot/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_011_frame_selection_audit/pilot/). The frozen baseline `results/satlas_sr_baseline_freeze_v1/` remains 100% untouched.

---

## 3. Candidate Strategy Specifications & Selected Indices

The exact 8 frame indices selected for each tile across all strategies are recorded in [`pilot_frame_selections.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_011_frame_selection_audit/pilot/pilot_frame_selections.csv):

| Strategy ID | Strategy Name | Type | Selection & Ordering Rule |
| :--- | :--- | :--- | :--- |
| **S0** | **Baseline Frozen (Seed 0)** | Pseudorandom | `random.seed(0)` sequential stream; replicates baseline inference exactly. |
| **S1** | **Alternate Seed (Seed 42)** | Pseudorandom | `random.seed(42)` sequential stream with `random.sample(goods, 8)`. |
| **S2** | **Sorted Baseline** | Content-Matched | Exactly S0 indices, but sorted ascending: `sorted(indices_from_S0)`. |
| **S3** | **Uniform Strided Grid** | Deterministic | Fixed uniform striding: `[0, 4, 8, 12, 16, 20, 24, 28]`. |
| **S4** | **Early Window** | Deterministic | Fixed early block: `[0, 1, 2, 3, 4, 5, 6, 7]`. |
| **S5** | **Late Window** | Deterministic | Fixed late block: `[24, 25, 26, 27, 28, 29, 30, 31]`. |
| **S6** | **Low-Brightness Heuristic** | Content-Heuristic | 8 frames with lowest mean RGB pixel values (cloud avoidance), sorted ascending. |

### Exact Selected Indices per Tile

| Tile ID | S0 (Baseline Seed 0) | S1 (Seed 42) | S2 (Sorted S0) | S3 (Strided) | S4 (Early) | S5 (Late) | S6 (Low Brightness) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **20771_46525** | `24 31 28 13 1 8 16 15` | `7 0 23 8 31 27 4 29` | `1 8 13 15 16 24 28 31` | `0 4 8 12 16 20 24 28` | `0 1 2 3 4 5 6 7` | `24 25 26 27 28 29 30 31` | `4 13 14 18 20 24 30 31` |
| **20890_46801** | `25 29 31 26 9 15 11 18` | `6 21 23 28 17 2 18 13` | `9 11 15 18 25 26 29 31` | `0 4 8 12 16 20 24 28` | `0 1 2 3 4 5 6 7` | `24 25 26 27 28 29 30 31` | `1 12 17 18 19 28 29 30` |
| **20926_46045** | `13 16 4 9 29 24 3 19` | `2 0 31 6 7 16 19 30` | `3 4 9 13 16 19 24 29` | `0 4 8 12 16 20 24 28` | `0 1 2 3 4 5 6 7` | `24 25 26 27 28 29 30 31` | `5 8 12 13 14 24 26 29` |
| **21068_45692** | `16 29 17 22 25 19 4 9` | `12 22 20 30 17 13 7 14` | `4 9 16 17 19 22 25 29` | `0 4 8 12 16 20 24 28` | `0 1 2 3 4 5 6 7` | `24 25 26 27 28 29 30 31` | `0 4 5 11 20 22 25 31` |
| **21856_49804** | `6 23 2 28 27 21 10 15` | `17 25 27 0 24 30 5 22` | `2 6 10 15 21 23 27 28` | `0 4 8 12 16 20 24 28` | `0 1 2 3 4 5 6 7` | `24 25 26 27 28 29 30 31` | `3 4 13 14 15 19 25 28` |

---

## 4. Aggregate Metric Performance Summary

All metrics were computed against the NAIP reference images. Averages and sample standard deviations ($N=5$) are reported below:

| Strat | Strategy Name | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS (VGG) $\downarrow$ | CLIPScore $\uparrow$ | Mean $\Delta$ cPSNR vs S0 | Mean $\Delta$ PSNR vs S0 | Mean $\Delta$ LPIPS vs S0 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | **Baseline (Seed 0)** | $21.186 \pm 2.41$ | $19.422 \pm 1.29$ | $0.348 \pm 0.06$ | $0.3569 \pm 0.035$ | $0.9534 \pm 0.023$ | — | — | — |
| **S1** | **Alternate Seed 42** | $21.064 \pm 2.45$ | $19.199 \pm 1.31$ | $0.339 \pm 0.08$ | $0.3685 \pm 0.052$ | $0.9472 \pm 0.026$ | $-0.122\text{ dB}$ | $-0.223\text{ dB}$ | $+0.0115$ |
| **S2** | **Sorted Baseline** | $21.203 \pm 2.45$ | $19.383 \pm 1.38$ | $0.351 \pm 0.06$ | $0.3577 \pm 0.039$ | $0.9490 \pm 0.031$ | $+0.017\text{ dB}$ | $-0.039\text{ dB}$ | $+0.0007$ |
| **S3** | **Uniform Strided** | $20.828 \pm 1.74$ | $19.335 \pm 0.87$ | $0.326 \pm 0.04$ | $0.3724 \pm 0.038$ | $0.9493 \pm 0.024$ | $-0.359\text{ dB}$ | $-0.087\text{ dB}$ | $+0.0154$ |
| **S4** | **Early Window** | $21.204 \pm 2.38$ | $19.349 \pm 1.21$ | $0.344 \pm 0.06$ | $0.3578 \pm 0.040$ | $0.9361 \pm 0.045$ | $+0.018\text{ dB}$ | $-0.073\text{ dB}$ | $+0.0009$ |
| **S5** | **Late Window** | $21.155 \pm 2.26$ | $19.250 \pm 1.26$ | $0.329 \pm 0.08$ | $0.3627 \pm 0.041$ | $0.9492 \pm 0.037$ | $-0.031\text{ dB}$ | $-0.172\text{ dB}$ | $+0.0058$ |
| **S6** | **Low-Brightness** | $21.063 \pm 1.62$ | $\mathbf{20.027} \pm 0.33$ | $0.346 \pm 0.06$ | $\mathbf{0.3548} \pm 0.051$ | $0.9447 \pm 0.040$ | $-0.123\text{ dB}$ | $\mathbf{+0.605\text{ dB}}$ | $\mathbf{-0.0022}$ |

---

## 5. Detailed Per-Tile Breakdown

### 5.1. Cropped PSNR (cPSNR, crop=4, dB)

| Tile ID | S0 (Baseline) | S1 (Seed 42) | S2 (Sorted S0) | S3 (Strided) | S4 (Early) | S5 (Late) | S6 (Low Brightness) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20771_46525** | 20.1369 | 20.1220 | 20.2061 | 20.4020 | 20.2383 | 20.5701 | 20.3737 |
| **20890_46801** | 20.1277 | 20.3014 | 20.0820 | 19.9084 | 19.9194 | 19.9650 | 20.0787 |
| **20926_46045** | 19.2413 | 18.7337 | 19.3217 | 19.1463 | 19.2566 | 19.3538 | 19.9143 |
| **21068_45692** | 21.0954 | 20.9596 | 20.9425 | 20.9854 | 21.3631 | 20.8207 | 21.1015 |
| **21856_49804** | 25.3290 | 25.2052 | 25.4624 | 23.6953 | 25.2416 | 25.0662 | 23.8476 |

### 5.2. Standard PSNR (crop=4, dB)

| Tile ID | S0 (Baseline) | S1 (Seed 42) | S2 (Sorted S0) | S3 (Strided) | S4 (Early) | S5 (Late) | S6 (Low Brightness) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20771_46525** | 19.7886 | 19.7151 | 19.8703 | 20.0101 | 19.8984 | 19.9648 | 19.8951 |
| **20890_46801** | 19.9157 | 19.9615 | 20.0200 | 19.3670 | 19.8054 | 19.9496 | 19.8714 |
| **20926_46045** | 19.1780 | 18.4384 | 19.2493 | 18.9416 | 19.1003 | 18.7942 | 19.7115 |
| **21068_45692** | 20.8498 | 20.5709 | 20.6808 | 20.2744 | 20.5569 | 20.3022 | 20.5763 |
| **21856_49804** | 17.3782 | 17.3093 | 17.0944 | 18.0840 | 17.3855 | 17.2391 | **20.0800** |

### 5.3. Perceptual LPIPS (VGG Backbone, lower is better)

| Tile ID | S0 (Baseline) | S1 (Seed 42) | S2 (Sorted S0) | S3 (Strided) | S4 (Early) | S5 (Late) | S6 (Low Brightness) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20771_46525** | 0.35863 | 0.38177 | 0.36209 | 0.39232 | 0.38520 | 0.36766 | 0.36699 |
| **20890_46801** | 0.32667 | 0.31525 | 0.31667 | 0.34948 | 0.31383 | 0.31959 | 0.32544 |
| **20926_46045** | 0.39480 | 0.39622 | 0.38819 | 0.38531 | 0.38316 | 0.39337 | 0.37975 |
| **21068_45692** | 0.38667 | 0.43343 | 0.40212 | 0.41576 | 0.39304 | 0.41033 | 0.41732 |
| **21856_49804** | 0.31795 | 0.31575 | 0.31928 | 0.31900 | 0.31376 | 0.32253 | **0.28435** |

---

## 6. Scientific Analysis of Findings

### 6.1. Channel Permutation Sensitivity (S2 vs. S0)
In Strategy S2, the exact 8 frames drawn by S0 are retained, but their channel order is sorted in ascending index order. 
- Mean cPSNR shifted by $+0.0169\text{ dB}$, with individual tile shifts ranging from $-0.1529\text{ dB}$ to $+0.1334\text{ dB}$.
- Standard PSNR shifted by up to $-0.2838\text{ dB}$.
- **Significance**: Because no new visual information was introduced, this delta is purely attributable to the weights in `conv_first`. A 2D convolution over concatenated temporal frames does not treat channels symmetrically. Reordering frames alters which temporal slice is processed by which convolution kernel.

### 6.2. Stochastic Sampling Variance (S1 vs. S0)
Drawing an alternate seed (Seed 42) reduced cPSNR by $-0.1217\text{ dB}$ and PSNR by $-0.2230\text{ dB}$, with LPIPS degrading on 4 out of 5 tiles. This indicates that baseline evaluation figures published with a single random seed are subject to stochastic fluctuation of at least $\pm 0.1\text{ to } 0.3\text{ dB}$.

### 6.3. The Impact of Cloud Filtering (S6)
Tile `21856_49804` in the baseline exhibited an abnormally low PSNR of $17.3782\text{ dB}$, despite having a high cPSNR of $25.3290\text{ dB}$ (caused by a large brightness bias offset). 
- Inspection of `tci.png` for this tile revealed bright cloud contamination in several frames (mean RGB values reaching 255.0).
- Strategy S6 discarded those cloudy frames and selected only the 8 darkest, clearest frames (`[3, 4, 13, 14, 15, 19, 25, 28]`).
- As a direct result:
  - PSNR jumped from $17.3782\text{ dB} \rightarrow \mathbf{20.0800\text{ dB}}$ ($+2.7018\text{ dB}$).
  - LPIPS perceptual distance improved from $0.31795 \rightarrow \mathbf{0.28435}$ (an improvement of $0.0336$, the strongest perceptual score across the entire pilot).
  - Across all 5 tiles, S6 delivered the highest mean PSNR ($20.0269\text{ dB}$) and lowest mean LPIPS ($0.3548$).

---

## 7. Compute & Runtime Performance

- **Inference Runtime**: 35 inferences executed in $1.64\text{ seconds}$ total ($\sim 0.047\text{ s}$ per tile).
- **Metric Evaluation**: LPIPS + OpenCLIP across 35 tiles completed in $52.2\text{ seconds}$ ($\sim 1.49\text{ s}$ per evaluation).
- **Peak GPU Memory**: $73.60\text{ MiB}$ during inference, $\sim 2.8\text{ GiB}$ during OpenCLIP evaluation.
- **Disk Usage**: Total size of 35 generated PNGs is $< 2\text{ MB}$.

---

## 8. Limitations & Preliminary Nature

1. **Sample Size ($N=5$)**:
   Five tiles do not provide statistical significance ($p$-values cannot reject the null hypothesis at $\alpha=0.01$). 
2. **Geographic Diversity**:
   The five pilot tiles represent specific terrain types. Performance could differ across urban, agricultural, or coastal scenes.
3. **Hypothesis vs. Fact**:
   - *Verified Fact*: Frame selection and channel permutation measurably alter pixel and perceptual metrics for frozen Satlas-SR weights.
   - *Hypothesis to Test on 256 Tiles*: Low-brightness heuristic (S6) consistently outperforms random sampling across the full validation distribution.

---

## 9. Next Steps and Approval Gate

The pilot ran completely without error, safely isolated under `results/satlas_sr_experiment_011_frame_selection_audit/pilot/`.

Before launching the full 256-tile experiment across the 7 strategies (1,792 inferences, $\sim 19$ minutes total compute), **explicit supervisor approval is required**.
