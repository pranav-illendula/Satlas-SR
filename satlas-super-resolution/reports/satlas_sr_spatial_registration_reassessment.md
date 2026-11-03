# Satlas-SR Spatial Registration Reassessment & Scientific Audit

- **Audit Date**: 2026-10-22
- **Audited Tiles**: 256 unique urban validation image pairs

### Executive Summary
A critical reassessment of the spatial registration diagnostic reveals that raw Fourier Phase Cross-Correlation on unwindowed image boundaries produced severe spectral wraparound artifacts on a subset of tiles (skewing the raw mean to 8.58 px and max to 82.22 px). When edge-attenuating Hanning windows are applied and cross-checked against Normalized Cross-Correlation (NCC):

1. **True Median Displacement**: On the reliable subset (214 / 256 tiles), the median displacement is **1.736 pixels** (IQR: [1.03, 3.434] px), with a mean of **3.863 ± 5.741 px**.
2. **Proportion Within Search Window**: **55.6%** of reliable tiles have displacement $\le 2$ pixels, and **86.9%** have displacement $\le 8$ pixels.
3. **cPSNR Compatibility**: The cPSNR search box ($[-8, +8]$ in $x$ and $y$) encompasses a maximum Euclidean displacement of $11.31$ px, accommodating virtually the entire reliable distribution.
4. **Scientific Limits**: Cross-correlation demonstrates spatial feature correlation, not absolute sensor georeferencing. Metric gains from cPSNR reflect robustness to translational and radiometric mismatch, not sensor alignment verification.
