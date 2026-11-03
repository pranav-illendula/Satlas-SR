# EXP-011: Full 256-Tile Sentinel-2 Frame-Selection Sensitivity Evaluation Report

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe` (PyTorch 2.1.0+cu121)  
**Hardware Platform**: Windows 11 Pro, NVIDIA RTX 2000 Ada Generation (16 GB VRAM)  
**Official Checkpoint**: [`experiments/esrgan_8S2.pth`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/experiments/esrgan_8S2.pth) (`params_ema`, SHA-256: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`)  
**Scope**: Full 256-Tile Validation Dataset ($N=256$, 7 Strategies, 1,792 Total Inferences)  
**Execution Status**: COMPLETED SUCCESSFULLY (100% Success, 1,792/1,792 Valid $128 \times 128$ PNGs Generated)  
**Timestamp**: 2026-10-22T22:31:00+05:30  

---

## 1. Executive Summary & Mandatory Preflight Resolution

This experiment presents the comprehensive 256-tile evaluation investigating whether alternative temporal observation selection strategies alter the super-resolution quality of the frozen Satlas-SR model (`SSR_RRDBNet`, 4× upscaling, 24 input channels).

### 1.1. Mandatory Preflight Check: Baseline Replication
Prior to inference, the candidate baseline strategy (`S0`) was verified against the frozen baseline manifest ([`results/satlas_sr_full_validation/inference_manifest.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv)):
- **Replication Score**: **256 / 256 exact matches (100.0%)** across all tiles.
- `random.seed(0)` evaluated across the alphanumerically sorted 256 tiles perfectly reproduced the identical 8-frame selections and channel order for every tile in the dataset.
- All 1,792 generated predictions across all 7 strategies were independently verified as valid $128 \times 128 \times 3$ uint8 RGB PNG images.

### 1.2. Primary Scientific Conclusions
1. **Primary Metric Invariance (cPSNR, crop border 4)**:  
   Across the full 256 tiles, **none** of the six alternative strategies produced a statistically significant difference in cPSNR compared to the baseline `S0` (all uncorrected paired $t$-test $p > 0.35$; all Bonferroni-corrected $p = 1.0$). Mean cPSNR deltas across all strategies fell within a narrow band of **$-0.0273\text{ dB}$ to $+0.0179\text{ dB}$**.
2. **Reconciliation with 5-Tile Pilot**:  
   The apparent large PSNR gain ($+0.605\text{ dB}$) observed in the 5-tile pilot for the low-brightness heuristic (`S6`) was driven by a single outlier tile (`21856_49804`) with severe cloud saturation. Across the full 256 tiles, `S6` raw PSNR actually decreased by $-0.2778\text{ dB}$ ($p = 0.0125$) due to shadow/underexposure bias, while its affine-corrected cPSNR remained virtually identical ($+0.0179\text{ dB}$, $p = 0.782$). This empirically underscores the necessity of full-distribution testing over pilot subsets.
3. **Channel Permutation Sensitivity (S2 vs. S0)**:  
   Sorting the exact same 8 frames into ascending order (`S2`) produced tile-level variances (standard deviation of delta $= 0.311\text{ dB}$), confirming that `SSR_RRDBNet`'s first convolution layer `conv_first` is non-commutative across channels. However, this effect is symmetric: 116 tiles improved while 140 worsened, resulting in a net population difference of only $-0.0179\text{ dB}$ ($p = 0.357$).
4. **Deterministic Uniform Striding as a Seed-Free Standard (S3)**:  
   Strategy `S3` (fixed uniform indices `[0, 4, 8, 12, 16, 20, 24, 28]`) completely eliminates random seeding and metadata dependencies while matching baseline cPSNR within $-0.0082\text{ dB}$ ($p = 0.882$) and improving LPIPS perceptual distance on 57.0% of tiles ($p = 0.071$).

---

## 2. Complete 256-Tile Results Matrix

All 1,792 predictions were evaluated against ground truth NAIP references ([`data/small_val_set/naip/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/data/small_val_set/naip/)) using the verified evaluation suite. Population means and sample standard deviations ($N=256$) are reported below:

| Strat | Strategy Name | Strategy Type | cPSNR (dB) $\uparrow$ [Primary] | Raw PSNR (dB) $\uparrow$ | SSIM (crop 4) $\uparrow$ | LPIPS (VGG) $\downarrow$ | CLIPScore $\uparrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | **Baseline Frozen (Seed 0)** | Pseudorandom | $20.2144 \pm 1.63$ | $18.8122 \pm 1.58$ | $0.3912 \pm 0.091$ | $0.3470 \pm 0.046$ | $0.9612 \pm 0.026$ |
| **S1** | **Alternate Seed (Seed 42)** | Pseudorandom | $20.1872 \pm 1.60$ | $18.7367 \pm 1.62$ | $0.3904 \pm 0.089$ | $0.3466 \pm 0.048$ | $0.9620 \pm 0.026$ |
| **S2** | **Sorted Baseline** | Permutation | $20.1965 \pm 1.64$ | $18.7981 \pm 1.57$ | $0.3906 \pm 0.091$ | $0.3471 \pm 0.046$ | $0.9606 \pm 0.027$ |
| **S3** | **Uniform Strided Grid** | Deterministic | $20.2062 \pm 1.61$ | $18.7812 \pm 1.61$ | $0.3930 \pm 0.090$ | $0.3452 \pm 0.047$ | $0.9615 \pm 0.026$ |
| **S4** | **Early Window** | Deterministic | $20.2305 \pm 1.60$ | $18.7957 \pm 1.61$ | $0.3909 \pm 0.091$ | $0.3470 \pm 0.047$ | $0.9617 \pm 0.026$ |
| **S5** | **Late Window** | Deterministic | $20.1899 \pm 1.63$ | $18.7416 \pm 1.60$ | $0.3875 \pm 0.091$ | $0.3466 \pm 0.047$ | $0.9614 \pm 0.026$ |
| **S6** | **Low-Brightness Heuristic** | Content-Heuristic | $20.2324 \pm 1.57$ | $18.5344 \pm 1.65$ | $0.3860 \pm 0.089$ | $0.3454 \pm 0.047$ | $0.9629 \pm 0.025$ |

*Note: For LPIPS, lower is better. For all other metrics, higher is better. Detailed per-tile values are archived in [`full_per_tile_metrics.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/full_per_tile_metrics.csv).*

---

## 3. Paired Statistical Hypothesis Testing vs. Baseline (S0)

For each alternative strategy ($S_k$), per-tile differences $\Delta_i = M_i(S_k) - M_i(S_0)$ were computed across all 256 tiles. Statistical significance was evaluated using two-tailed Paired Student's $t$-tests and Wilcoxon signed-rank tests, with Bonferroni correction for 6 simultaneous family-wise comparisons ($M=6$):

### 3.1. Primary Fidelity Metric: cPSNR (crop border 4, dB)

| Strat | Mean $\Delta$ | Median $\Delta$ | 95% Confidence Interval | Win / Loss (%) | Paired $t$-stat | Uncorrected $p$ | Bonferroni $p$ | Wilcoxon $p$ | Significant? ($\alpha=0.01$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | $-0.0273\text{ dB}$ | $-0.0325\text{ dB}$ | $[-0.1368, +0.0822]$ | 118 / 138 (46.1%) | $-0.491$ | $0.624$ | $1.000$ | $0.620$ | **NO** |
| **S2** | $-0.0179\text{ dB}$ | $-0.0146\text{ dB}$ | $[-0.0562, +0.0204]$ | 116 / 140 (45.3%) | $-0.922$ | $0.357$ | $1.000$ | $0.305$ | **NO** |
| **S3** | $-0.0082\text{ dB}$ | $+0.0287\text{ dB}$ | $[-0.1180, +0.1015]$ | 134 / 122 (52.3%) | $-0.148$ | $0.882$ | $1.000$ | $0.343$ | **NO** |
| **S4** | $+0.0161\text{ dB}$ | $-0.0063\text{ dB}$ | $[-0.0904, +0.1226]$ | 128 / 128 (50.0%) | $+0.297$ | $0.766$ | $1.000$ | $0.615$ | **NO** |
| **S5** | $-0.0246\text{ dB}$ | $-0.0118\text{ dB}$ | $[-0.1302, +0.0811]$ | 126 / 130 (49.2%) | $-0.458$ | $0.647$ | $1.000$ | $0.926$ | **NO** |
| **S6** | $+0.0179\text{ dB}$ | $+0.0683\text{ dB}$ | $[-0.1096, +0.1454]$ | 150 / 106 (58.6%) | $+0.277$ | $0.782$ | $1.000$ | $0.035$ | **NO** |

### 3.2. Secondary & Perceptual Metrics Summary

| Strategy & Metric | Mean $\Delta$ | 95% CI | Win / Loss (%) | Paired $t$ $p$-val | Wilcoxon $p$-val | Bonferroni $p$ | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 (Seed 42)** Raw PSNR | $-0.0756\text{ dB}$ | $[-0.241, +0.090]$ | 132 / 124 (51.6%) | $0.369$ | $0.826$ | $1.000$ | No |
| **S1 (Seed 42)** LPIPS (VGG) $\downarrow$ | $-0.0004$ | $[-0.0037, +0.0029]$ | 130 / 126 (50.8%) | $0.817$ | $0.712$ | $1.000$ | No |
| **S2 (Sorted)** Raw PSNR | $-0.0141\text{ dB}$ | $[-0.075, +0.047]$ | 115 / 141 (44.9%) | $0.650$ | $0.415$ | $1.000$ | No |
| **S2 (Sorted)** LPIPS (VGG) $\downarrow$ | $+0.0001$ | $[-0.0011, +0.0014]$ | 119 / 137 (46.5%) | $0.826$ | $0.572$ | $1.000$ | No |
| **S3 (Strided)** Raw PSNR | $-0.0310\text{ dB}$ | $[-0.208, +0.146]$ | 141 / 115 (55.1%) | $0.730$ | $0.359$ | $1.000$ | No |
| **S3 (Strided)** LPIPS (VGG) $\downarrow$ | $-0.0018$ | $[-0.0049, +0.0013]$ | 146 / 110 (57.0%) | $0.260$ | $0.071$ | $0.427$ | No |
| **S6 (Low Bright)** Raw PSNR | $-0.2778\text{ dB}$ | $[-0.495, -0.060]$ | 117 / 139 (45.7%) | $0.0125$ | $0.048$ | $0.075$ | Marg. ($p < 0.05$ uncorr) |
| **S6 (Low Bright)** LPIPS (VGG) $\downarrow$ | $-0.0015$ | $[-0.0055, +0.0025]$ | 145 / 111 (56.6%) | $0.458$ | $0.083$ | $0.496$ | No |
| **S6 (Low Bright)** CLIPScore $\uparrow$ | $+0.0017$ | $[-0.0005, +0.0040]$ | 132 / 124 (51.6%) | $0.136$ | $0.547$ | $0.819$ | No |

*Full statistical test records are archived in [`paired_statistical_tests.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/paired_statistical_tests.csv).*

---

## 4. In-Depth Technical & Scientific Analysis

### 4.1. Reconciliation: Why the 5-Tile Pilot Diverged from the 256-Tile Reality
In the 5-tile pilot:
- `S6` showed an extraordinary PSNR gain of $+0.605\text{ dB}$ on average, powered by tile `21856_49804` jumping $+2.70\text{ dB}$ ($17.38 \rightarrow 20.08\text{ dB}$).
- This pilot tile contained extreme white cloud saturation across several frames. By selecting darker frames, `S6` effectively prevented severe cloud blowout.

Across the full 256 tiles:
- While `S6` indeed rescued a handful of similarly cloudy tiles, for the vast majority of clear or partially shadowed terrain chips, picking strictly the 8 darkest frames introduced an **under-exposure / shadow bias**.
- Consequently, raw PSNR for `S6` dropped by **$-0.2778\text{ dB}$** across the population ($p = 0.0125$).
- However, when evaluated with **cPSNR** (which performs spatial translation and affine brightness adjustment to neutralize global illumination shifts), `S6` achieved $20.2324\text{ dB}$ ($+0.0179\text{ dB}$ delta, $58.6\%$ of tiles improved).
- **Core Lesson**: Unsupervised brightness heuristics do not function as true cloud masks. Selecting darker frames trades cloud avoidance for shadow capture. Small pilot evaluations can easily produce deceptive outlier-driven signals.

### 4.2. Channel Permutation Sensitivity (S2 vs. S0)
In Strategy `S2`, the exact 8 observations drawn by `S0` were preserved, but sorted into chronological index order prior to channel concatenation:
- At the individual tile level, reordering channels caused substantial divergence: tile deltas ranged from $-0.95\text{ dB}$ to $+0.84\text{ dB}$ with a standard deviation of $0.311\text{ dB}$.
- This confirms the architectural reality: `SSR_RRDBNet`'s input layer (`nn.Conv2d(24, 64)`) assigns independent learned kernel weights to each channel. Concatenating frames in different permutations changes early feature maps.
- However, at the population scale, there is **zero systematic directional benefit** to sorting: the mean delta is $-0.0179\text{ dB}$ ($p = 0.357$). Because the network was trained with randomized draws, its weights learned to be broadly resilient to channel permutations without favoring ascending order.

### 4.3. Uniform Striding (S3) as an Engineering Recommendation
Strategy `S3` selects frames at fixed stride 4: `[0, 4, 8, 12, 16, 20, 24, 28]`.
- **Zero Stochasticity**: Requires no PRNG seed, ensuring 100% deterministic reproducibility regardless of tile execution order.
- **Performance**:
  - cPSNR: $20.2062\text{ dB}$ vs Baseline $20.2144\text{ dB}$ ($\Delta = -0.0082\text{ dB}$, $p = 0.882$).
  - LPIPS: $0.3452$ vs Baseline $0.3470$ (improves perceptual quality on 57.0% of tiles).
  - SSIM: $0.3930$ vs Baseline $0.3912$.
- **Recommendation**: In production or future benchmarking pipelines, replacing random sampling with uniform deterministic striding (`S3`) eliminates seed ambiguity with zero penalty to super-resolution fidelity.

---

## 5. Execution Metrics & Resource Profiling

| Stage | Scale | Total Time | Throughput | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Preflight Verification** | 256 tiles, 5 checks | 3.2s | — | Minimal |
| **Inference (7 Strategies)** | 1,792 inferences | 76.83s | **23.32 tiles/sec** | 73.6 MiB |
| **NAIP Precomputation** | 256 reference arrays + CLIP embeds | 14.8s | 17.3 tiles/sec | 2.8 GiB |
| **Metric Suite Evaluation** | 1,792 predictions (5 metrics each) | 843.24s | 2.13 tiles/sec | 2.8 GiB |
| **Statistical Analysis** | 30 hypothesis tests + Bonferroni | 2.1s | — | Minimal |
| **Total Experiment Runtime** | End-to-end | **1,044.0s (17.4 min)** | — | **2.8 GiB** |

---

## 6. Scientific Limitations & Boundaries

1. **Exploratory Sensitivity Evidence Only**:
   These findings demonstrate that the existing Satlas-SR checkpoint is robust to observation selection within the curated 32-frame buffer. This is an evaluation of **pipeline sensitivity**, not a claim that the model weights have improved or that higher resolution has been unlocked.
2. **Pre-Curated Validation Set**:
   In `small_val_set`, 100% of the 32 frames per tile are valid (`goods=32`, `bads=0`). In raw, uncurated operational satellite archives, severe nodata borders and heavy cloud coverage may occur. While `S6` demonstrated that cloud filtering rescues severe outliers, a true cloud detector requires spectral mask bands (e.g. Sentinel-2 Scene Classification Layer / QA60), which are absent from this dataset.
3. **Historical Baseline Integrity**:
   The frozen baseline metrics published in earlier reports remain completely valid and intact. Baseline `S0` achieved $20.2144\text{ dB}$ cPSNR, exactly replicating the official benchmark.
