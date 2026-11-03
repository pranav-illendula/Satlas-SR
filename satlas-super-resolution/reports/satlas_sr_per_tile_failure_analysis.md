# Satlas-SR Per-Tile Failure and Disagreement Analysis

This report examines the 256 validation tiles to identify systematic failure modes, rank extreme tiles, and analyze cases where evaluation metrics fundamentally disagree.

## 1. Ranked Extreme Tiles

### Top 5 Highest cPSNR (Best Fidelity)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `31350_48832` | 39.54 | 28.94 | 0.9129 | 0.2077 | 0.9838 | 8.56 | UNRELIABLE_LOW_CORRELATION |
| 2 | `35674_55337` | 36.96 | 34.78 | 0.8762 | 0.1898 | 0.8420 | 15.40 | UNRELIABLE_EXTREME_OUTLIER |
| 3 | `33866_53889` | 32.98 | 25.59 | 0.7258 | 0.1824 | 0.9828 | 0.92 | UNRELIABLE_LOW_CORRELATION |
| 4 | `33818_53875` | 32.73 | 27.01 | 0.6533 | 0.3441 | 0.9695 | 15.23 | UNRELIABLE_LOW_CORRELATION |
| 5 | `37564_49521` | 31.85 | 30.59 | 0.8794 | 0.2504 | 0.9690 | 11.14 | RELIABLE |

### Bottom 5 Lowest cPSNR (Severe Pixel Mismatch)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `35816_54902` | 9.59 | 7.49 | 0.0953 | 0.5930 | 0.8898 | 1.82 | UNRELIABLE_LOW_CORRELATION |
| 2 | `27328_49676` | 10.58 | 7.48 | 0.1866 | 0.5592 | 0.8765 | 0.50 | RELIABLE |
| 3 | `24893_52620` | 10.66 | 8.82 | 0.1347 | 0.4897 | 0.9725 | 1.60 | UNRELIABLE_LOW_CORRELATION |
| 4 | `26634_53065` | 12.04 | 11.01 | 0.2768 | 0.5749 | 0.8662 | 2.16 | UNRELIABLE_LOW_CORRELATION |
| 5 | `24873_52643` | 13.53 | 5.60 | 0.0501 | 0.6626 | 0.8796 | 60.31 | UNRELIABLE_EXTREME_OUTLIER |

### Top 5 Lowest LPIPS (Best Perceptual Realism)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `35771_54925` | 30.69 | 28.83 | 0.8346 | 0.1442 | 0.9458 | 2.38 | RELIABLE |
| 2 | `33453_48667` | 22.41 | 21.77 | 0.6812 | 0.1814 | 0.9720 | 31.11 | RELIABLE |
| 3 | `33866_53889` | 32.98 | 25.59 | 0.7258 | 0.1824 | 0.9828 | 0.92 | UNRELIABLE_LOW_CORRELATION |
| 4 | `35674_55337` | 36.96 | 34.78 | 0.8762 | 0.1898 | 0.8420 | 15.40 | UNRELIABLE_EXTREME_OUTLIER |
| 5 | `35747_54922` | 20.83 | 20.61 | 0.6016 | 0.2006 | 0.9763 | 2.41 | RELIABLE |

### Top 5 Highest LPIPS (Severe Perceptual Artifacts)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `34477_54104` | 17.57 | 15.23 | 0.1514 | 0.7196 | 0.8766 | 3.04 | RELIABLE |
| 2 | `28398_50891` | 19.89 | 11.48 | 0.2195 | 0.6907 | 0.7768 | 10.82 | UNRELIABLE_LOW_CORRELATION |
| 3 | `24873_52643` | 13.53 | 5.60 | 0.0501 | 0.6626 | 0.8796 | 60.31 | UNRELIABLE_EXTREME_OUTLIER |
| 4 | `30541_51662` | 23.48 | 15.19 | 0.2919 | 0.6058 | 0.9201 | 20.58 | SPECTRAL_WRAPAROUND_CORRECTED |
| 5 | `35816_54902` | 9.59 | 7.49 | 0.0953 | 0.5930 | 0.8898 | 1.82 | UNRELIABLE_LOW_CORRELATION |

### Top 5 Highest CLIPScore (Strong Semantic Content)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `37893_48603` | 15.38 | 15.46 | 0.3655 | 0.3188 | 0.9930 | 1.20 | RELIABLE |
| 2 | `35677_55330` | 17.71 | 17.62 | 0.5241 | 0.2648 | 0.9929 | 0.92 | RELIABLE |
| 3 | `32877_49734` | 16.74 | 16.04 | 0.3546 | 0.3031 | 0.9926 | 1.72 | RELIABLE |
| 4 | `31359_48853` | 19.45 | 19.07 | 0.4649 | 0.2643 | 0.9924 | 0.10 | RELIABLE |
| 5 | `27387_50159` | 14.78 | 14.67 | 0.3451 | 0.3191 | 0.9918 | 0.72 | RELIABLE |

### Bottom 5 Lowest CLIPScore (Low Semantic Clarity)

| Rank | Tile ID | cPSNR (dB) | PSNR (dB) | SSIM | LPIPS | CLIPScore | Shift (px) | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `28398_50891` | 19.89 | 11.48 | 0.2195 | 0.6907 | 0.7768 | 10.82 | UNRELIABLE_LOW_CORRELATION |
| 2 | `27295_49506` | 23.29 | 20.27 | 0.6473 | 0.3048 | 0.8129 | 3.81 | RELIABLE |
| 3 | `35674_55337` | 36.96 | 34.78 | 0.8762 | 0.1898 | 0.8420 | 15.40 | UNRELIABLE_EXTREME_OUTLIER |
| 4 | `35674_55425` | 20.18 | 18.13 | 0.2421 | 0.3241 | 0.8524 | 2.11 | RELIABLE |
| 5 | `27375_49421` | 15.26 | 12.62 | 0.5326 | 0.3767 | 0.8567 | 5.14 | UNRELIABLE_LOW_CORRELATION |

## 2. Representative Disagreement Analysis

Evaluation metrics often diverge due to different mathematical objectives:

### Case A: High cPSNR with Poor Perceptual Realism (High LPIPS)
Occurs when the model generates smooth, low-contrast imagery that closely matches regional pixel averages (minimizing MSE) but lacks crisp building boundaries or natural aerial texture.

### Case B: Low cPSNR with High Semantic Similarity (High CLIPScore)
Occurs in complex urban tiles with slight translational misregistration or minor seasonal tree canopy change. While pixel-wise MSE suffers severely, the high-level semantic features (roads, buildings, vegetation) remain recognized by the vision foundation model.

### Case C: Massive cPSNR Gain (> 4 dB)
Demonstrates cases where translation search and brightness bias correction successfully compensate for sensor calibration offsets between Sentinel-2 and aerial NAIP.

## 3. Visual Contact Sheets

Contact sheets showing low-resolution input, SR model prediction, and aerial NAIP reference are saved under `results/satlas_sr_revalidation/visualizations_v2/`:

### Best cPSNR (`31350_48832`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/best_cpsnr_31350_48832.png`

### Worst cPSNR (`35816_54902`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/worst_cpsnr_35816_54902.png`

### Best Perceptual LPIPS (`35771_54925`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/best_lpips_35771_54925.png`

### Worst Perceptual LPIPS (`34477_54104`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/worst_lpips_34477_54104.png`

### Highest CLIPScore (`37893_48603`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/best_clip_37893_48603.png`

### Lowest CLIPScore (`28398_50891`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/worst_clip_28398_50891.png`

### Massive cPSNR Gain (Translation/Bias) (`38409_49923`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/massive_gain_38409_49923.png`

### Disagreement: High cPSNR / High LPIPS (`31350_48832`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/high_cpsnr_poor_lpips_31350_48832.png`

### Disagreement: Low cPSNR / High CLIP (`24893_52620`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/low_cpsnr_high_clip_24893_52620.png`

### Outlier: High Displacement / Artifact (`24873_52643`)
- Path: `results/satlas_sr_revalidation/visualizations_v2/severe_outlier_24873_52643.png`

