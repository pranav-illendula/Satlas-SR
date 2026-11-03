# Satlas-SR Paper-Aligned Full Validation Evaluation Report

- **Date (UTC)**: 2026-10-21T18:09:49.243269+00:00
- **Evaluated Tiles**: 256 / 256 (100.0% success)
- **Device**: NVIDIA RTX 2000 Ada Generation (cuda)
- **Predictions**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\results\satlas_sr_full_validation`
- **References**: `C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set\naip`

## 1. Aggregate Paper Metric Results

| Metric | Mean ± Std | Min | Max | Denominator | Configuration / Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cPSNR (crop 4)** | **20.2144 ± 4.1478 dB** | 9.5912 | 39.5373 | 256 | `ssr/metrics/cpsnr.py`, max_offset=8, crop_border=4 |
| **cPSNR (no crop)** | **20.1111 ± 4.1132 dB** | 9.7854 | 39.5008 | 256 | `ssr/metrics/cpsnr.py`, max_offset=8, crop_border=0 |
| **LPIPS (VGG)** | **0.347 ± 0.0809** | 0.1442 | 0.7196 | 256 | `ssr/metrics/lpips.py`, VGG backbone, raw distance (lower is better) |
| **CLIPScore** | **0.9612 ± 0.0344** | 0.7768 | 0.993 | 256 | `open_clip`, `clipa-ViT-bigG-14`, cosine similarity |
| **PSNR (crop 4)** | 18.887 ± 3.86 dB | 5.5999 | 34.7798 | 256 | BasicSR `calculate_psnr`, crop_border=4 |
| **PSNR (raw/full)** | 18.8122 ± 3.8138 dB | 5.726 | 33.8205 | 256 | skimage full-frame array PSNR |
| **SSIM (crop 4)** | 0.3912 ± 0.1564 | 0.0501 | 0.9129 | 256 | BasicSR `calculate_ssim`, crop_border=4 |
| **SSIM (raw/full)** | 0.395 ± 0.1495 | 0.057 | 0.9059 | 256 | skimage full-frame array SSIM |

## 2. Key Analysis: Standard PSNR vs. cPSNR

- Standard PSNR (crop border 4) averages **18.887 dB**, while translation/bias-optimized **cPSNR (crop border 4)** averages **20.2144 dB** (mean improvement: **+1.33 dB**).
- This substantial increase demonstrates that sub-pixel registration shifts and atmospheric/illumination radiometric biases account for a major portion of the raw pixel error in remote sensing super-resolution, exactly as discussed in the Satlas-SR paper.

## 3. Five Pilot Tiles Breakdown

| Tile ID | cPSNR (crop 4) | LPIPS (VGG) | CLIPScore | PSNR (crop 4) | SSIM (crop 4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `20771_46525` | 20.1369 dB | 0.35863 | 0.94874 | 19.7886 dB | 0.28959 |
| `20890_46801` | 20.1277 dB | 0.32667 | 0.97549 | 19.9157 dB | 0.41776 |
| `20926_46045` | 19.2413 dB | 0.3948 | 0.96007 | 19.178 dB | 0.31919 |
| `21068_45692` | 21.0954 dB | 0.38667 | 0.91549 | 20.8498 dB | 0.30606 |
| `21856_49804` | 25.329 dB | 0.31795 | 0.96705 | 17.3782 dB | 0.40977 |
