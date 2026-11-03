# Evaluation & Benchmarking Suite

The `evaluation/` directory contains tools and metrics for evaluating super-resolved satellite imagery against high-resolution reference datasets (such as NAIP).

---

## Metric Definitions

### 1. Cyclic Peak Signal-to-Noise Ratio (cPSNR)
Derived from the European Space Agency's PROBA-V Super-Resolution Challenge. Because satellite platforms experience subtle orbital vibrations, attitude perturbations, and sensor alignment jitter, rigid pixel-by-pixel PSNR heavily penalizes sub-pixel spatial shifts.
- **Formulation**:
  $$\text{cPSNR} = \max_{\Delta x, \Delta y, \mathbf{b}} 10 \log_{10}\left(\frac{255^2}{\text{MSE}\left(I_{\text{pred}}(\Delta x, \Delta y) + \mathbf{b}, I_{\text{ref}}\right)}\right)$$
- Evaluates an 81-point translation grid with $\pm 8$ pixel radius and per-channel brightness bias compensation $\mathbf{b}$.
- Border crop: 4 pixels (`crop_border=4`).

### 2. Learned Perceptual Image Patch Similarity (LPIPS)
Computes the Euclidean distance between deep feature representations extracted from a pretrained VGG-16 backbone. Correlates strongly with human visual perception and surface realism compared to pixel L1/L2 losses.
- Bounded in $[0, \infty)$ where lower scores indicate greater perceptual fidelity.

### 3. CLIPScore (CLIPA ViT-bigG-14-336)
Measures high-level visual semantic consistency using the vision transformer of OpenCLIP ViT-bigG-14-CLIPA-336.
- Bounded within $[0.70, 1.00]$ for natural satellite scenes.
- Verifies that reconstructed features (roads, buildings, agricultural fields) retain correct semantic context.

### 4. Standard PSNR & SSIM
- **Standard PSNR**: Evaluated with a 4-pixel border crop.
- **SSIM**: Structural similarity calculated with a 7x7 Gaussian window (`data_range=255.0`).

---

## Usage Examples

### 1. Evaluate Prediction Directory (Standard Metrics)
```bash
python evaluation/evaluate_validation.py \
  --predictions satlas-super-resolution/results/satlas_sr_pilot_five_tiles \
  --dataset data/small_val_set \
  --output-csv results/validation_report.csv
```

### 2. Compute Official Paper Metrics Benchmark
```bash
python evaluation/evaluate_paper_metrics.py \
  --predictions satlas-super-resolution/results/satlas_sr_full_validation \
  --dataset data/small_val_set \
  --output-json results/paper_metrics.json \
  --output-csv results/paper_metrics.csv
```

### 3. Non-Parametric Bootstrap Statistical Analysis
Calculate 95% bootstrap confidence intervals, median, and interquartile distribution stats:
```bash
python evaluation/run_statistical_analysis.py satlas-super-resolution/results/satlas_sr_revalidation/per_tile_metrics.csv
```

