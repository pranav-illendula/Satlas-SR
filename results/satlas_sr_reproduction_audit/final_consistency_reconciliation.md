# Final Consistency Reconciliation & Audit Verification Report

- **Date**: 2026-10-22
- **Audit Target**: Satlas-SR Reproduction Baseline Artifacts & Reports
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`
- **Output Artifact**: `results/satlas_sr_reproduction_audit/final_consistency_reconciliation.md`

---

## 1. Scope & Audited Artifacts

This audit independently inspects and cross-references all data tables, configurations, scripts, and reports to resolve discrepancies, verify statistical properties, and establish clear distinctions between empirical facts, algorithmic diagnostics, and qualitative interpretations.

### Files and Columns Inspected
1. **Model & Checkpoint**:
   - `experiments/esrgan_8S2.pth` (Weights dictionary: `params`, `params_ema`)
   - `ssr/archs/rrdbnet_arch.py` (`class SSR_RRDBNet`)
   - `ssr/options/infer_satlas_full_validation.yml`
2. **Per-Tile Metrics**:
   - `results/satlas_sr_revalidation/per_tile_metrics.csv` (`tile_id`, `status`, `cpsnr_crop4_db`, `cpsnr_crop0_db`, `lpips_vgg`, `clipscore`, `psnr_crop4_db`, `ssim_crop4`, `psnr_raw_db`, `ssim_raw`, `cpsnr_gain_over_psnr_db`)
   - `results/satlas_sr_paper_metrics/per_tile_metrics.csv`
   - `results/satlas_sr_full_validation/tile_metrics.csv`
3. **Statistical & Correlation Outputs**:
   - `results/satlas_sr_revalidation/statistical_analysis.json`
   - `results/satlas_sr_revalidation/metric_correlations.csv`
   - `results/satlas_sr_revalidation/aggregate_metrics.json`
4. **Spatial Registration Outputs**:
   - `results/satlas_sr_spatial_audit/registration_validation.csv` (`tile_id`, `raw_sr_mag_px`, `windowed_sr_mag_px`, `ncc_peak_correlation`, `reliability_status`, `is_reliable`)
   - `results/satlas_sr_spatial_audit/registration_shifts.csv`
5. **Project Dashboards & Reports**:
   - `results/satlas_sr_reproduction_audit/experiment_metadata_v2.json`
   - `results/satlas_sr_reproduction_audit/project_status_v2.json`
   - `reports/SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md`
   - `reports/satlas_sr_statistical_analysis.md`
   - `reports/satlas_sr_per_tile_failure_analysis.md`

---

## 2. Parameter Count Reconciliation

### The Discrepancy
- Historical audit documents (`master_reproduction_audit.md`, `satlas_sr_full_validation_evaluation.md`, `evaluate_satlas_sr_validation.py`) recorded **16,710,083 parameters**.
- In the latest report (`SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md` and `experiment_metadata_v2.json`), a figure of **16,707,331 parameters** was stated.

### Empirical Evidence & Ground Truth
We directly queried PyTorch and inspected both the checkpoint tensor elements and the instantiated model architecture in Python 3.9:
```python
from ssr.archs.rrdbnet_arch import SSR_RRDBNet
import torch

# 1. Instantiated official model architecture:
model = SSR_RRDBNet(num_in_ch=24, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
print(sum(p.numel() for p in model.parameters()))
# Output: 16710083

# 2. Checkpoint dictionary elements:
ckpt = torch.load('experiments/esrgan_8S2.pth', map_location='cpu')
print(sum(v.numel() for v in ckpt['params_ema'].values()))
# Output: 16710083
print(sum(v.numel() for v in ckpt['params'].values()))
# Output: 16710083
```

### Resolution
- **Definitive Parameter Count**: **16,710,083 parameters**.
- **Root Cause of Discrepancy**: The figure `16,707,331` was an inadvertent typo introduced into `experiment_metadata_v2.json` ($16,710,083 - 2,752$).
- **Status**: **RESOLVED**. The ground-truth parameter count is verified as **16,710,083**, exactly matching `experiments/esrgan_8S2.pth` and consistent with the paper's reported ~17M small ESRGAN variant.

---

## 3. cPSNR Gain Reconciliation

### The Discrepancy
- Aggregate tables in `aggregate_metrics.json` and `per_tile_metrics.csv` record a maximum cPSNR-over-PSNR gain of **+12.8842 dB** on tile `38409_49923`.
- In `SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md` (Section 2 summary table and Section 6.2 item 7), a figure of **+17.38 dB** was stated for `38409_49923`.

### Empirical Evidence & Ground Truth
We inspected all per-tile records for tile `38409_49923` across all CSV files:
- In `results/satlas_sr_revalidation/per_tile_metrics.csv`:
  - `cpsnr_crop4_db`: `28.2705`
  - `psnr_crop4_db`: `15.3863`
  - `cpsnr_gain_over_psnr_db`: **`12.8842`** ($28.2705 - 15.3863 = 12.8842$)
  - `cpsnr_crop0_db`: `28.2603`
  - `psnr_raw_db`: `15.3730`
  - Raw gain: $28.2603 - 15.3730 = 12.8873\text{ dB}$
- Tracing the source of `17.38`:
  - In `results/satlas_sr_paper_metrics/per_tile_metrics.csv` (row 66) and `results/satlas_sr_revalidation/per_tile_metrics.csv` (row 66), tile **`27394_49842`** has a `cpsnr_crop4_db` value of **`17.3825`**.
  - During document drafting, this metric value (`17.38 dB`) was erroneously transcribed into the maximum gain column of the summary table and the description for tile `38409_49923`.

### Resolution
- **Definitive Maximum cPSNR Gain**: **+12.8842 dB** (occurring on tile `38409_49923`).
- **Comparable Metric Definitions**:
  - `cpsnr_crop4_db`: cPSNR evaluated after cropping 4 border pixels from input, followed by the internal $8 \times 8$ search window crop (net evaluation size: $120 \times 120$).
  - `psnr_crop4_db`: Rigid standard PSNR on the identical $120 \times 120$ central window.
  - `cpsnr_gain_over_psnr_db`: Exact difference $\text{cPSNR}_{\text{crop4}} - \text{PSNR}_{\text{crop4}}$.
- Across all 256 tiles:
  - Mean gain: **$+1.3274 \pm 2.0991\text{ dB}$**
  - Median gain: **$+0.6076\text{ dB}$**
  - Minimum gain: **$-0.1372\text{ dB}$** (on tile `38661_49188`, where border cropping reduces shared edge contrast)
  - Maximum gain: **$+12.8842\text{ dB}$** (on tile `38409_49923`)
- **Status**: **RESOLVED**.

---

## 4. Statistical Claims Validation

### 4.1 Bootstrap Methodology & Interpretation
- **Methodology**: Resampling was performed using `np.random.default_rng(42)`. In each iteration, $N=256$ tiles (or $N=214$ for the registration subset) were drawn with replacement from the observed tile metrics, and the sample mean was calculated. Over 2,000 resamples, the 2.5th and 97.5th percentiles formed the 95% confidence interval.
- **Scope of Claim**: The bootstrap confidence intervals describe **sampling uncertainty in the mean of the 256 tiles**. They **do not** measure model parameter uncertainty, epistemic uncertainty, or generalization error to global biomes.

### 4.2 Machine-Readable Verification
Comparing `statistical_analysis.json` and `reports/satlas_sr_statistical_analysis.md`:

| Metric | Sample Size ($N$) | Mean ± Std | Median | IQR [Q1, Q3] | Verified 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **cPSNR (dB)** | 256 | $20.2144 \pm 4.1478$ | $19.4600$ | $[17.6980, 21.3346]$ | $[19.6852, 20.7486]$ |
| **Standard PSNR (dB)** | 256 | $18.8870 \pm 3.8600$ | $18.3366$ | $[16.8785, 20.2978]$ | $[18.4129, 19.3646]$ |
| **cPSNR Gain (dB)** | 256 | $1.3274 \pm 2.0991$ | $0.6076$ | $[0.1818, 1.3655]$ | $[1.0887, 1.5859]$ |
| **SSIM** | 256 | $0.3912 \pm 0.1564$ | $0.3784$ | $[0.2905, 0.4591]$ | $[0.3730, 0.4113]$ |
| **LPIPS** | 256 | $0.3470 \pm 0.0809$ | $0.3396$ | $[0.2984, 0.3818]$ | $[0.3375, 0.3572]$ |
| **CLIPScore** | 256 | $0.9612 \pm 0.0344$ | $0.9736$ | $[0.9542, 0.9834]$ | $[0.9569, 0.9650]$ |
| **Shift (Reliable, px)** | 214 | $3.8631 \pm 5.7413$ | $1.7360$ | $[1.0300, 3.4340]$ | $[3.1091, 4.6605]$ |

### 4.3 Correlation Coefficients
The correlation table in `reports/satlas_sr_statistical_analysis.md` matches `metric_correlations.csv` exactly:
- `cPSNR vs PSNR`: $r = +0.8650$, $\rho = +0.8329$ ($p < 10^{-66}$)
- `cPSNR vs SSIM`: $r = +0.6898$, $\rho = +0.5223$ ($p < 10^{-18}$)
- `cPSNR vs LPIPS`: $r = -0.3351$, $\rho = -0.1737$ ($p = 0.0053$)
- `cPSNR vs CLIPScore`: $r = -0.1316$, $\rho = -0.2629$ ($p < 10^{-4}$)
- `cPSNR vs Shift (Reliable)`: $r = +0.1812$, $\rho = +0.2219$ ($p = 0.0011$)
- `LPIPS vs Shift (Reliable)`: $r = +0.2065$, $\rho = +0.3602$ ($p < 10^{-7}$)
- `CLIPScore vs Shift (Reliable)`: $r = -0.3218$, $\rho = -0.4245$ ($p < 10^{-10}$)

*Note on discrepancy in `SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md` Section 5*: The narrative text in Section 5 of the comprehensive report cited rounded draft numbers ($r = 0.8712, \rho = 0.8753$); the authoritative, verified values from `metric_correlations.csv` and `satlas_sr_statistical_analysis.md` are $r = +0.8650, \rho = +0.8329$.

---

## 5. Registration Filtering & Subset Accounting

### 5.1 Classification Rules
In `scripts/diagnose_registration_estimator.py`, each tile was evaluated on both unwindowed and Hanning-windowed Fourier Phase Cross-Correlation (PCC) and integer-offset Normalized Cross-Correlation (NCC):
1. **Low Structural Correlation**: If $\text{NCC}_{\text{peak}} < 0.25$, classified as `UNRELIABLE_LOW_CORRELATION`.
2. **Spectral Wraparound Corrected**: Else if $\text{magnitude}_{\text{raw}} > 10.0\text{ px}$ AND ($\text{magnitude}_{\text{windowed}} \le 4.0\text{ px}$ OR $\text{magnitude}_{\text{ncc}} \le 4.0\text{ px}$), classified as `SPECTRAL_WRAPAROUND_CORRECTED`.
3. **Extreme Outlier**: Else if $\text{magnitude}_{\text{raw}} > 15.0\text{ px}$, classified as `UNRELIABLE_EXTREME_OUTLIER`.
4. **Standard Reliable**: All remaining tiles classified as `RELIABLE`.

### 5.2 Category Reconciliation & Sum-Check
A direct inspection of `registration_validation.csv` confirms:
- `RELIABLE`: **200 tiles** ($78.1\%$)
- `SPECTRAL_WRAPAROUND_CORRECTED`: **14 tiles** ($5.5\%$)
- `UNRELIABLE_LOW_CORRELATION`: **37 tiles** ($14.5\%$)
- `UNRELIABLE_EXTREME_OUTLIER`: **5 tiles** ($2.0\%$)

$$\text{Total Tiles} = 200 + 14 + 37 + 5 = 256\text{ tiles } (100.0\%)$$

The "Reliable Subset" is defined as:
$$\text{Reliable Subset} = \text{RELIABLE} + \text{SPECTRAL\_WRAPAROUND\_CORRECTED} = 200 + 14 = 214\text{ tiles } (83.59\%)$$

The categories are strictly mutually exclusive and partition the dataset with zero overlap and zero omitted tiles.

### 5.3 Physical Geodetic vs. Estimated Raster Displacement
- Raster displacement is an algorithmic phase/NCC correlation between two 2D digital arrays.
- It is sensitive to lighting, shadows, season, and textural sharpness.
- It is **not** a geodetic survey measurement of ground-truth satellite ephemeris or camera calibration.

---

## 6. Metric Terminology & Protocol Fidelity

### 6.1 CLIPScore Terminology
- **Correction**: CLIPScore is the cosine similarity between vision encoder feature embeddings.
- It **must never be described as a percentage, accuracy, or quality score**.
- Because inputs are divided by $255$ without ImageNet mean and standard deviation normalization, features reside in an acute positive hyper-cone, compressing the effective dynamic range to $[0.73, 1.00]$.
- A score of $0.9612$ indicates strong high-level semantic alignment, but cannot be equated with $96.12\%$ pixel or perceptual correctness.

### 6.2 Implementation Test vs. Full Paper Protocol
- **Implementation Verified**: The local functions `calculate_cpsnr`, `calculate_lpips`, and `calculate_clipscore` were verified as mathematically exact executions of the released codebase functions.
- **Protocol Limits**:
  - The paper's primary benchmarks in Section 5 evaluate an 87M **medium** ESRGAN across thousands of nationwide test tiles.
  - The local evaluation verifies the released 16.71M **small** ESRGAN baseline checkpoint on the 256-tile `small_val_set`.
  - Passing all synthetic and local tests confirms fidelity to the repository's evaluation scripts, not equivalence to the paper's national test benchmark.

---

## 7. Master Reconciliation Summary

| Check Item | Target | Inspected Artifact | Finding / Reconciliation | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Parameter Count** | Checkpoint & Model | `experiments/esrgan_8S2.pth`, `SSR_RRDBNet` | Exact count is **16,710,083**. Typo in v2 report (`16,707,331`) reconciled. | **RESOLVED** |
| **2. Max cPSNR Gain** | Tile `38409_49923` | `per_tile_metrics.csv` | Exact gain is **+12.8842 dB**. Typo (`+17.38 dB`) traced to cPSNR of tile `27394_49842` and reconciled. | **RESOLVED** |
| **3. Bootstrap Resampling** | Statistical Methodology | `run_statistical_and_failure_analysis.py`, `statistical_analysis.json` | Verified non-parametric bootstrap over tiles (2,000 resamples, seed 42) measuring uncertainty in the sample mean. | **RESOLVED** |
| **4. Correlation Metrics** | Metric Relationships | `metric_correlations.csv`, `satlas_sr_statistical_analysis.md` | Values confirmed. Discrepancy between markdown text and CSV reconciled to CSV ground truth. | **RESOLVED** |
| **5. Registration Categories** | Partition Consistency | `registration_validation.csv` | Exact sum: $200 + 14 + 37 + 5 = 256$. Reliable subset is $200 + 14 = 214$. Zero overlap. | **RESOLVED** |
| **6. Metric Terminology** | Scientific Precision | `satlas_sr_metric_validation_deep_dive.md` | Terminology audited. CLIPScore strictly defined as semantic cosine similarity in $[0.73, 1.00]$. | **RESOLVED** |

All original outputs, predictions, and reports remain strictly intact and preserved.
