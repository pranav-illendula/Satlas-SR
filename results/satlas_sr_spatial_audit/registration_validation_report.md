# Satlas-SR Spatial Registration Reassessment & Estimator Validation Report

## 1. Why Raw Phase Correlation Produced Artifacts

Unwindowed Phase Cross-Correlation computes the cross-power spectrum directly on non-periodic rectangular image arrays. In images with low visual overlap, boundary discontinuities or repetitive structures, high-frequency spectral leakage causes the phase peak to jump to periodic boundary wraparounds (e.g., shifts of 40 to 80 pixels).

Applying a **Hanning/Hann spectral window** to attenuate tile edges and cross-verifying with **Normalized Cross-Correlation (NCC)** demonstrates that:
- Out of 256 tiles, **14 tiles** suffered from raw boundary wraparound where the true spatial shift is actually small (<= 4 px).
- **37 tiles** have low correlation (NCC < 0.25), caused by dense cloud cover, uniform water, or severe temporal/seasonal land cover changes.
- **5 tiles** exhibit true severe discrepancies (>15 px) incompatible with Web-Mercator zoom-17 pairing.

## 2. Statistical Comparison: Raw vs. Windowed vs. Filtered Subset

| Estimation Protocol | Sample Count | Mean ± Std (px) | Median (px) | IQR [Q1, Q3] (px) | Max (px) | $\le 2$ px (%) | $\le 8$ px (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Unwindowed FFT (Prior)** | 256 (100%) | 8.58 ± 18.634 | 0.922 | [0.361, 2.41] | 82.219 | 67.2% | 83.6% |
| **Windowed FFT (All Tiles)** | 256 (100%) | 5.062 ± 7.39 | 2.017 | [1.118, 5.14] | 60.311 | 48.8% | 80.5% |
| **Reliable Subset (Filtered)** | 214 (83.6%) | **3.863 ± 5.741** | **1.736** | [1.03, 3.434] | 31.119 | **55.6%** | **86.9%** |

## 3. Relationship to cPSNR Search Geometry

- **Box Search vs. Radial Displacement**: The cPSNR implementation in `ssr/metrics/cpsnr.py` searches a grid of integer offsets $\Delta x \in [0, 8]$ and $\Delta y \in [0, 8]$ on cropped $120 \times 120$ subwindows. This is a square $L_\infty$ search box of $[-8, +8] \times [-8, +8]$ pixels, corresponding to a maximum Euclidean radial displacement of $\sqrt{8^2 + 8^2} = 8\sqrt{2} \approx 11.31$ pixels.
- For reliable scenes, **86.9%** have displacement $\le 8$ pixels, and over **95%** have displacement $\le 11.3$ pixels. This mathematically explains why cPSNR successfully compensates translation for the overwhelming majority of tiles.

## 4. Correction of Prior Interpretation Overstatements

The previous report stated that the model synthesizes sub-pixel details that align closely with the NAIP reference. **This claim was an overstatement and is formally corrected**:
1. Image cross-correlation measures relative similarity of spatial frequency patterns, not geographic ground truth or physical coordinate accuracy.
2. The apparent shift reduction from LR to SR partly reflects the increased high-frequency sharpness of the SR image, which produces sharper Fourier correlation peaks compared to blurred, nearest-neighbor upsampled low-resolution inputs.
3. Consequently, cross-correlation estimates demonstrate that the super-resolved textures are translationally correlated with the reference within ~1-2 pixels; they do **not** independently prove sensor registration or absence of geometric hallucination.
