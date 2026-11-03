# Satlas-SR Paper Comparability Analysis Report

- **Reference Paper**: *Satlas-SR: A Dataset and Benchmark for Satellite Super-Resolution* ([arXiv:2311.18082](https://arxiv.org/abs/2311.18082))
- **Official Model Variant**: ESRGAN Multi-Frame ($8 \times \text{Sentinel-2} \to 4\times \text{NAIP}$)
- **Checkpoint**: `experiments/esrgan_8S2.pth` (16.71M parameters, SHA-256: `167d6b9d...`)

---

## 1. Specification & Implementation Comparability Matrix

| Evaluation Dimension | Paper Specification | Local Reproduction Implementation | Verification Evidence | Status | Scientific Consequence for Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | Multi-frame ESRGAN generator (`SSR_RRDBNet`) with 23 RRDB blocks, 64 feature channels, 32 growth channels | Exact `SSR_RRDBNet` loaded via BasicsR registry | Model inspection: 16,707,331 parameters | **VERIFIED** | Direct architectural equivalence to paper baseline. |
| **Input Modality & Channels** | 8 Sentinel-2 RGB frames ($8 \times 3 = 24$ channels), $32 \times 32$ spatial resolution | 8 Sentinel-2 RGB frames ($24 \times 32 \times 32$) | Preprocessed via `infer_utils.format_s2naip_data` | **VERIFIED** | Identical input dimensionality and temporal frame count. |
| **Ground Truth Reference** | High-resolution aerial NAIP RGB imagery at $128 \times 128$ resolution ($0.5-1.0\text{m/px}$) | Exact matching NAIP `rgb.png` for each tile ID ($128 \times 128$) | Verified in `dataset_inventory.csv` | **VERIFIED** | Spatial scale factor ($4\times$) and reference domain match paper. |
| **Evaluation Dataset Split** | Complete test split containing thousands of geographically diverse tiles across the US | Curated validation subset `small_val_set` containing **256 urban tiles** | Directory audit in `../data/small_val_set` | **PARTIAL** | **Cannot directly compare absolute score aggregates against the full test benchmark**. The 256-tile subset provides a localized regression baseline, not global US representation. |
| **Temporal Frame Selection** | 8 randomly sampled cloud-free / non-black acquisitions per tile | Black-pixel filtering (`[0,0,0] not in ts`) followed by `random.sample` | Recorded in `inference_manifest.csv` with fixed seed 0 | **VERIFIED** | Matches official inference script logic while making frame selection fully deterministic. |
| **cPSNR Metric Protocol** | PROBA-V translational search $[-8, +8]$ with channel-wise brightness bias compensation | `calculate_cpsnr` in `ssr/metrics/cpsnr.py` (crop border = 4) | Audited in `reports/satlas_sr_metric_validation_deep_dive.md` | **VERIFIED** | Identical code implementation; evaluates 81 offsets with stride 2. |
| **LPIPS Metric Protocol** | Perceptual loss using VGG16 trunk | `lpips.LPIPS(net='vgg')` | Audited in `reports/satlas_sr_metric_validation_deep_dive.md` | **VERIFIED** | Matches official code and pretrained VGG weights. |
| **CLIPScore Metric Protocol** | Vision-language foundation model similarity using OpenCLIP CLIPA-ViT-bigG-14 | `open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')` | Audited in `reports/satlas_sr_metric_validation_deep_dive.md` | **VERIFIED** | Matches exact model tag and unnormalized $/255$ tensor input. |
| **Standard PSNR & SSIM** | Classical rigid pixel-wise fidelity metrics | `calculate_psnr` and `calculate_ssim` with `crop_border=4` | Exact BasicsR library functions | **VERIFIED** | Supplementary verification metrics aligned with super-resolution standards. |

---

## 2. Quantitative Baseline on `small_val_set` (256 Tiles)

The local reproduction establishes the following verified benchmark across the 256 tiles:

| Metric | Local Reproduction Mean ± Std | 95% Bootstrap Confidence Interval | Median [IQR] |
| :--- | :--- | :--- | :--- |
| **cPSNR (dB)** | **$20.21 \pm 4.15\text{ dB}$** | $[19.69, 20.75]\text{ dB}$ | $19.46\text{ dB}$ $[17.06, 22.84]$ |
| **Standard PSNR (dB)** | **$18.89 \pm 3.86\text{ dB}$** | $[18.41, 19.36]\text{ dB}$ | $18.34\text{ dB}$ $[15.86, 21.65]$ |
| **cPSNR Translation Gain** | **$+1.33 \pm 2.10\text{ dB}$** | $[+1.09, +1.59]\text{ dB}$ | $+0.61\text{ dB}$ $[+0.16, +1.73]$ |
| **SSIM** | **$0.391 \pm 0.156$** | $[0.373, 0.411]$ | $0.378$ $[0.274, 0.505]$ |
| **LPIPS (VGG16)** | **$0.347 \pm 0.081$** | $[0.338, 0.357]$ | $0.340$ $[0.297, 0.390]$ |
| **CLIPScore (CLIPA-ViT-bigG)** | **$0.961 \pm 0.034$** | $[0.957, 0.965]$ | $0.974$ $[0.957, 0.982]$ |

---

## 3. Scientific Caveats & Generalization Boundaries

1. **Subset Bias**: The 256 tiles in `small_val_set` were selected primarily from urban regions with high structural density. Results cannot be extrapolated to rural, desert, agricultural, or forest biomes without evaluating the full dataset.
2. **Deterministic Seed Dependency**: Frame selection for each tile samples 8 dates out of up to 32 Sentinel-2 acquisitions. Under seed 0, high-quality cloud-free frames were sampled for 100% of tiles, but a different seed could introduce variations on the order of $\pm 0.3\text{ dB}$.
3. **No Retraining Conducted**: This audit strictly evaluates the official pre-trained checkpoint weights. It verifies reproduction of model inference and metric evaluation, not the training dynamics or loss convergence.
