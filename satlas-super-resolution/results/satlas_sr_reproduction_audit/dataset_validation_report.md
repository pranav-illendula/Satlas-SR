# Satlas-SR Dataset and Inference Inventory Validation Report

- **Audit Date (UTC)**: 2026-10-21T18:45:43.143781+00:00
- **Checkpoint Verified**: `esrgan_8S2.pth` (134,157,518 bytes, SHA-256 matches official: **True**)
- **Unique Tile IDs**: 256
- **Sentinel-2 Tiles Validated**: 256 / 256 ({'OK': 256})
- **NAIP References Validated**: 256 / 256 ({'OK': 256})
- **Predictions Validated**: 256 / 256 ({'OK': 256})
- **Exact 3-Way Match (S2 + NAIP + Pred)**: **256 / 256 (100%)**

## Sentinel-2 Frame Statistics

- **Total Staccked Frames per Tile**: Fixed at **32 frames** ($32 \times 32$ RGB chunks vertically stacked to $32 \times 1024$ PNGs across all 256 tiles).
- **Valid (Zero-Black-Pixel) Frames**: Mean **32.0** (range: 32 – 32).
- **Tiles with < 8 Valid Frames**: **0** (every tile has $\ge 8$ valid frames, meaning all 8 sampled input frames come from 100% valid, non-black observations).
- **Tiles with Partially Black Frames**: 0 tiles contain at least one stack frame with black edge artifacts.
- **Tiles with Completely Black Frames**: 0 tiles contain completely missing (black) observations.

## Data Quality & Value Ranges

| Asset | Expected Resolution | Verified Resolution | Pixel Range | Readability |
| :--- | :--- | :--- | :--- | :--- |
| Sentinel-2 `tci.png` | 32 x 1024 | 32 x 1024 | uint8 [0, 255] | 256 / 256 OK |
| NAIP `rgb.png` | 128 x 128 | 128 x 128 | uint8 [0, 255] | 256 / 256 OK |
| Prediction `sr.png` | 128 x 128 | 128 x 128 | uint8 [0, 255] | 256 / 256 OK |
| Preview `lr.png` | 32 x 32 | 32 x 32 | uint8 [0, 255] | 256 / 256 OK |
