# Satlas-SR Statistical Analysis & Metric Correlation Report

- **Dataset**: 256 paired urban validation tiles
- **Methodology**: Non-parametric bootstrap resampling (2,000 resamples, fixed seed = 42)
- **Confidence Intervals**: Two-tailed 95% percentile bootstrap [2.5%, 97.5%]

## 1. Aggregate Metric Summary & 95% Confidence Intervals

| Metric | Sample Count | Mean ± Std | Median | IQR [Q1, Q3] | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cPSNR (dB)** | 256 | 20.2144 ± 4.1478 | 19.46 | [17.698, 21.3346] | **[19.6852, 20.7486]** |
| **Standard PSNR (dB)** | 256 | 18.887 ± 3.86 | 18.3366 | [16.8785, 20.2978] | **[18.4129, 19.3646]** |
| **SSIM** | 256 | 0.3912 ± 0.1564 | 0.3784 | [0.2905, 0.4591] | **[0.373, 0.4113]** |
| **LPIPS (VGG)** | 256 | 0.347 ± 0.0809 | 0.3396 | [0.2984, 0.3818] | **[0.3375, 0.3572]** |
| **CLIPScore (CLIPA-ViT-bigG)** | 256 | 0.9612 ± 0.0344 | 0.9736 | [0.9542, 0.9834] | **[0.9569, 0.965]** |
| **cPSNR Gain (dB)** | 256 | 1.3274 ± 2.0991 | 0.6076 | [0.1818, 1.3655] | **[1.0887, 1.5859]** |
| **Estimated Shift (px, Reliable Subset)** | 214 | 3.8631 ± 5.7413 | 1.736 | [1.03, 3.434] | **[3.1091, 4.6605]** |

## 2. Metric Pairwise Correlation Matrix

| Metric Pair | N | Pearson $r$ | Pearson $p$-value | Spearman $\rho$ | Spearman $p$-value | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPSNR vs PSNR** | 256 | +0.8650 | 4.9603e-78 | **+0.8329** | 3.3043e-67 | Strong positive |
| **CPSNR vs SSIM** | 256 | +0.6898 | 1.7299e-37 | **+0.5223** | 2.5749e-19 | Moderate positive |
| **CPSNR vs LPIPS** | 256 | -0.3351 | 3.891e-08 | **-0.1737** | 0.0053333 | Uncorrelated negative |
| **CPSNR vs CLIPSCORE** | 256 | -0.1316 | 0.035272 | **-0.2629** | 2.0299e-05 | Weak negative |
| **CPSNR vs SHIFT_PX** | 214 | +0.1812 | 0.0078749 | **+0.2219** | 0.0010819 | Weak positive |
| **PSNR vs SSIM** | 256 | +0.6548 | 9.9267e-33 | **+0.5138** | 1.1987e-18 | Moderate positive |
| **PSNR vs LPIPS** | 256 | -0.5430 | 4.9615e-21 | **-0.3755** | 5.4187e-10 | Weak negative |
| **PSNR vs CLIPSCORE** | 256 | +0.0465 | 0.4588 | **-0.0708** | 0.25907 | Uncorrelated negative |
| **PSNR vs SHIFT_PX** | 214 | +0.1109 | 0.10566 | **+0.0822** | 0.23101 | Uncorrelated positive |
| **SSIM vs LPIPS** | 256 | -0.5197 | 4.1425e-19 | **-0.5084** | 3.1078e-18 | Moderate negative |
| **SSIM vs CLIPSCORE** | 256 | -0.0181 | 0.77282 | **-0.0633** | 0.31272 | Uncorrelated negative |
| **SSIM vs SHIFT_PX** | 214 | +0.0531 | 0.43994 | **-0.1172** | 0.087306 | Uncorrelated negative |
| **LPIPS vs CLIPSCORE** | 256 | -0.4886 | 9.1e-17 | **-0.4179** | 3.0546e-12 | Moderate negative |
| **LPIPS vs SHIFT_PX** | 214 | +0.2065 | 0.0023946 | **+0.3602** | 5.9212e-08 | Weak positive |
| **CLIPSCORE vs SHIFT_PX** | 214 | -0.3218 | 1.52e-06 | **-0.4245** | 8.9733e-11 | Moderate negative |

## 3. Scientific Methodology & Sample Limitations

1. **Nature of Bootstrap Confidence Intervals**: The 95% bootstrap confidence intervals quantify statistical sampling uncertainty over the 256 tiles drawn from this specific small validation set. They **do not** measure model parameter uncertainty, nor do they prove geographic generalization across unseen biomes or global distributions.
2. **Independence Assumption**: The 256 tiles represent discrete geographic urban centers. While geographically dispersed, satellite orbits introduce spatial autocorrelation and seasonal clustering across adjacent flight lines.
3. **Disagreement Between Fidelity and Perceptual Metrics**: cPSNR and SSIM correlate moderately ($ho = +0.655$), but LPIPS shows weak negative rank correlation with cPSNR ($ho = -0.320$). This occurs because cPSNR rewards spatial smoothness and luminance matching, whereas LPIPS rewards sharp high-frequency edge textures that may diverge from the pixel-wise ground truth.
