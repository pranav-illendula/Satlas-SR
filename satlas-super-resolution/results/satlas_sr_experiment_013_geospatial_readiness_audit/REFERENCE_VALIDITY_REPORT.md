# EXP-013: Reference Validity, Geospatial Comparability, and Building-Level Readiness Report

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Python Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Audit Focus**: Task B (Geospatial & Temporal Comparability) and Task C (Reference Validity for Building Mapping)  
**Audit Status**: COMPLETED  
**Date**: 2026-10-23  

---

## 1. Executive Summary

This report evaluates whether the NAIP reference imagery and Sentinel-2 inputs in the local Satlas-SR evaluation dataset (`..\data\small_val_set`) support scientifically defensible claims regarding **spatial registration accuracy, temporal comparability, and building-level super-resolution reconstruction**.

### Definitive Conclusions:
1. **Four Concepts Formally Distinguished**:
   - Matching tile IDs $\neq$ Visual correspondence $\neq$ Estimated image displacement $\neq$ Verified geographic registration.
   - Matching folder names (`X_Y`) establish only that two data streams were extracted for the same nominal Web Mercator grid cell. They do not prove that individual pixels represent identical geographic coordinates on the ground.
2. **Spatial Registration Evidence & Limitations**:
   - The prior registration audit (`results/satlas_sr_spatial_audit/`) established that on a filtered **reliable subset of 214 tiles (83.6%)**, relative image displacement between super-resolved predictions and NAIP reference has a median of **1.74 pixels** (mean $3.86 \pm 5.74\text{ px}$).
   - **42 tiles (16.4%)** fail reliable registration: 37 suffer from low visual correlation (NCC peak $< 0.25$) caused by clouds, open water, or major seasonal/temporal land-cover changes, and 5 exhibit severe spatial shifts ($> 15\text{ px}$).
   - Cross-correlation measures **relative spatial texture alignment**, not geodetic ground truth. It cannot verify whether the NAIP reference itself or the Sentinel-2 input has absolute geodetic accuracy.
3. **Temporal Asynchrony & Sensor Confounding**:
   - NAIP reference images were captured during aerial surveys in 2019 (April to December). The 32 Sentinel-2 frames have zero recorded acquisition timestamps. Differences between SR outputs and NAIP references inevitably reflect legitimate temporal land-use changes, building construction/demolition, phenological changes, and shadow geometry, rather than model reconstruction error alone.
4. **Building Reference Validity**:
   - **Zero building annotations, polygons, or masks exist locally**.
   - NAIP RGB imagery cannot serve as a building ground truth without explicit building semantic labels. Standard pixel metrics (cPSNR, SSIM, LPIPS) measure whole-scene RGB radiance matching, which is heavily dominated by background vegetation, soil, and roads.
   - **No building-level accuracy or structural reliability claims can be made using the local dataset alone**.

---

## 2. Four Crucial Distinctions in Geospatial Super-Resolution

In satellite image super-resolution literature, four distinct concepts are frequently conflated. For scientific rigor, they are defined and separated below:

```
[Level 1: Nominal Tile Pairing]
       │  (Matching folder names X_Y in Slippy Map grid)
       ▼
[Level 2: Visual Correspondence]
       │  (Human or algorithmic perception of recognizable landmarks)
       ▼
[Level 3: Estimated Image Displacement]
       │  (Phase cross-correlation / NCC translation vector [dx, dy])
       ▼
[Level 4: Verified Geographic Registration]
          (Sub-pixel geodetic ground truth tied to surveyed benchmarks)
```

1. **Matching Tile Identities (Level 1)**:
   - *Definition*: Identical string identifiers (e.g., `20771_46525`) designating a tile cell in the EPSG:3857 Web Mercator hierarchy.
   - *What it establishes*: The pipeline targeted the same bounding box during tile extraction.
   - *What it does NOT establish*: It does not ensure that the satellite pushbroom sensor and aerial frame camera captured the scene without geometric distortion, terrain parallax, or georeferencing offsets.
2. **Visual Correspondence (Level 2)**:
   - *Definition*: Qualitative recognition that prominent spatial features (coastlines, arterial roads, large structures) appear in both images.
   - *What it establishes*: The images depict the same general landscape.
   - *What it does NOT establish*: Does not provide a numerical bound on spatial displacement, nor does it guarantee that fine structures (e.g., individual houses) align at the single-pixel scale.
3. **Estimated Image-to-Image Displacement (Level 3)**:
   - *Definition*: Shift vectors $[\Delta x, \Delta y]$ estimated via Normalized Cross-Correlation (NCC) or Hann-windowed Phase Cross-Correlation (PCC).
   - *What it establishes*: The rigid 2D translation that maximizes mutual spectral/spatial correlation between two raster arrays.
   - *What it does NOT establish*: It does not establish absolute geodetic coordinates. A zero shift merely indicates that dominant image edges align; if both images share a systematic geolocation error, or if one image is distorted by terrain elevation (orthorectification tilt), cross-correlation cannot detect it.
4. **Verified Geographic Registration (Level 4)**:
   - *Definition*: Rigorous geodetic verification against surveyed Ground Control Points (GCPs) or high-precision geodetic basemaps with sub-meter Root Mean Square Error (RMSE).
   - *Current Status*: **Completely absent in the Satlas-SR evaluation set**.

---

## 3. Audit of Existing Registration Diagnostics

The existing spatial registration audit conducted in [`results/satlas_sr_spatial_audit/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_spatial_audit/) was reviewed to assess what evidence is established.

### 3.1. Reassessment of Phase Correlation and Spectral Windowing

- **Spectral Leakage Diagnosis**: Unwindowed 2D Fourier phase correlation suffers from high-frequency spectral leakage when applied to non-periodic satellite tiles. Discontinuities at image borders cause phase correlation peaks to jump to opposite edges (producing spurious shift estimates of 40 to 80 pixels).
- **Hann Windowing**: Applying a 2D Hann window attenuates tile boundary discontinuities, suppressing edge artifacts and isolating interior feature alignment.
- **Empirical Findings** (from [`registration_validation_report.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_spatial_audit/registration_validation_report.md)):
  - **Full Dataset ($N=256$)**: Windowed FFT displacement mean is $5.06 \pm 7.39\text{ px}$, median $2.02\text{ px}$.
  - **Reliable Subset ($N=214$, 83.6%)**: Excluding low-correlation scenes yields a displacement of **$3.86 \pm 5.74\text{ px}$** (median **$1.74\text{ px}$**, IQR $[1.03, 3.43]\text{ px}$). Within this subset, **86.9% of tiles have displacement $\le 8$ pixels**.
  - **Flagged Unreliable Subset ($N=42$, 16.4%)**:
    * **37 tiles** exhibit low Normalized Cross-Correlation (peak $\text{NCC} < 0.25$). Inspection reveals dense cloud cover, open featureless water, or severe temporal land-use change.
    * **5 tiles** exhibit true severe displacement ($> 15\text{ px}$), indicating severe coordinate or extraction anomalies incompatible with direct pixel comparison.

### 3.2. What Registration Diagnostics Can and Cannot Establish

| Diagnostic Capability | Established by Diagnostics? | Explanation / Technical Reality |
| :--- | :---: | :--- |
| Relative 2D translation between SR and NAIP | **YES** | Windowed FFT and NCC robustly quantify bulk rigid translation for scenes with adequate texture. |
| Compensation capability of cPSNR metric | **YES** | The cPSNR metric searches a $\pm 8\text{ px}$ window ($11.31\text{ px}$ Euclidean diagonal). Over 95% of reliable tiles fall within this search box. |
| Absolute geodetic location accuracy | **NO** | Relative correlation cannot distinguish whether the satellite or the aircraft was geodetically true. |
| Non-rigid local deformation / parallax | **NO** | Bulk 2D Fourier shifts model only uniform planar translation. Elevation-dependent building parallax (building lean caused by off-nadir viewing) cannot be detected. |
| Sub-pixel building boundary alignment | **NO** | Building edges represent high-frequency details that contribute minimally to bulk Fourier phase compared to regional roads and field boundaries. |

---

## 4. Geospatial and Temporal Comparability Audit

### 4.1. Spatial Footprint and Pixel Grid Alignment

- **Coordinate System**: Both Sentinel-2 inputs and NAIP references are provided on the EPSG:3857 Web Mercator grid at zoom level 17.
- **Grid Compatibility**:
  - The $128 \times 128$ NAIP reference raster exactly covers the spatial extent of the $32 \times 32$ Sentinel-2 raster at a $4\times$ nominal pixel ratio ($128 / 32 = 4$).
  - However, because the original sensors (Sentinel-2 MSI in UTM and NAIP frame camera in UTM/State Plane) have different native projections and pixel sizes ($10\text{ m}$ vs $0.6 - 1.0\text{ m}$), both datasets were bilinearly or bicubically resampled into Web Mercator.
  - This resampling introduces smoothing and sub-pixel phase shifts that cannot be reversed.

### 4.2. Temporal Comparability and Seasonality

- **NAIP Reference Timeline**: 239 USGS quadrangle directories record flight dates strictly between **2019-04-12 and 2019-12-15**. NAIP represents a single, near-instantaneous aerial snapshot during leaf-on conditions in 2019.
- **Sentinel-2 Timeline**: The 32 Sentinel-2 frames possess **no timestamps**. They may span months or multiple years (e.g., 2019–2021).
- **Physical Disparities Resulting from Temporal Gaps**:
  1. *New Construction / Demolition*: Buildings constructed after the NAIP flight date will appear in Sentinel-2 but not in NAIP, causing high reconstruction error that is not model failure.
  2. *Vegetation Phenology*: Seasonal canopy changes obscure building rooflines in summer and expose them in winter.
  3. *Sun Angle and Shadow Shifts*: Different acquisition times alter solar azimuth and elevation, radically shifting roof shadows. In high-resolution aerial imagery, shadows define building boundaries; mismatched shadows create large pixel residuals.

---

## 5. Reference Validity for Building-Level Evaluation

### 5.1. Audit of Building Footprints and Annotations

A comprehensive scan of the repository was conducted for building-related annotation files:
- Search patterns: `*building*`, `*mask*`, `*footprint*`, `*osm*`, `*.geojson`, `*.shp`.
- **Result**: Zero local vector or raster building masks exist.
- Training configuration files (`satlas_sr_config.json`) contain remote references to `osm_chips_to_masks.json`, but these files were never downloaded or included in the local repository or `small_val_set`.

### 5.2. RGB Reconstruction Metrics vs. Building Structural Metrics

A fundamental flaw in evaluating building mapping with whole-scene RGB metrics is the conflation of radiometric reconstruction with structural validity:

| Metric Category | Specific Metrics | What They Measure | Relevance to Building Mapping |
| :--- | :--- | :--- | :--- |
| **RGB Pixel Fidelity** | cPSNR, Raw PSNR, MSE, MAE | Whole-image photometric similarity in RGB space. | **Very Low**: Heavily dominated by large background areas (grass, asphalt, soil). A model can hallucinate or omit a small building with negligible change in PSNR ($< 0.1\text{ dB}$). |
| **Perceptual Similarity** | SSIM, LPIPS, CLIPScore | Patch-level textural and deep feature correspondence. | **Low**: Sensitive to edge contrast and texture realism, but penalizes valid sharp reconstructions if slightly translated, and rewards hallucinated realistic-looking building textures. |
| **Structural / Boundary Fidelity** | Boundary F1 (BF1), Hausdorff Distance, Corner Distance | Geometric alignment of reconstructed building edges to true building walls. | **Essential**: Direct measure of building outline fidelity. **Cannot be computed without vector/raster building masks.** |
| **Semantic Completeness** | Mask IoU, Precision, Recall | Overlap between detected building regions and true footprints. | **Essential**: Measures whether buildings are detected, omitted, or falsely invented. **Cannot be computed without ground truth masks.** |

```
┌────────────────────────────────────────────────────────────────────────┐
│ Why Whole-Image cPSNR Fails as a Proxy for Building Reliability:       │
│                                                                        │
│ In a 128x128 tile (16,384 pixels), a typical rural/suburban building   │
│ of 10m x 10m occupies ~4x4 = 16 pixels (0.098% of the scene).           │
│                                                                        │
│ Total scene pixels: 16,384 px                                          │
│ Building pixels:        16 px  (0.10% of total area)                   │
│ Background pixels:  16,368 px  (99.90% of total area)                  │
│                                                                        │
│ Even a total hallucination or complete omission of the building        │
│ alters whole-image MSE by less than 0.1%, undetectable in cPSNR!       │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.3. Required Reference Data for Defensible Building Evaluation

To evaluate reconstructed building details scientifically in a future study, the following external reference datasets would be mandatory:
1. **Authoritative Building Vector Footprints**: Co-registered polygon shapefiles (e.g., Microsoft US Building Footprints or OpenStreetMap building layers) filtered for temporal validity in 2019.
2. **Co-Registered High-Resolution Binary Masks**: Rasterized masks aligned with the NAIP pixel grid to evaluate IoU, Precision, Recall, and Boundary F1.
3. **Temporal Alignment**: Ground-truth vectors verified against 2019 aerial photography to prevent penalizing models for post-2019 construction.
4. **Geodetic Tie Points**: Sub-meter surveyed control points to ensure that sub-pixel building offsets represent model error rather than global tile registration jitter.

---

## 6. Permissible vs. Impermissible Claims

Based on the evidence audited in Task B and Task C, clear boundaries are established for what can and cannot be claimed:

### Permissible Claims (Evidence-Grounded):
- *Claim 1*: Sentinel-2 inputs and NAIP references share the same nominal Slippy Map zoom-17 coordinate cell across all 256 tiles.
- *Claim 2*: On 214 of 256 tiles (83.6%), image features exhibit strong cross-correlation with a median relative 2D translation of 1.74 pixels.
- *Claim 3*: The cPSNR translation search window ($\pm 8\text{ px}$) compensates for the relative translation observed across >95% of reliable tiles.
- *Claim 4*: 42 tiles exhibit severe spatial discrepancies or low cross-correlation and must be quarantined from fine-scale benchmark evaluations.

### Impermissible Claims (Unsubstantiated / Scientifically Unsound):
- *Forbidden Claim 1*: "The Satlas-SR model produces geographically accurate building reconstructions." (Zero building ground-truth annotations exist; geodetic accuracy is unverified).
- *Forbidden Claim 2*: "NAIP imagery provides absolute ground truth for evaluating Sentinel-2 super-resolution." (Temporal gaps, differing sensor physics, and off-nadir perspective lean prevent NAIP from acting as absolute ground truth).
- *Forbidden Claim 3*: "High cPSNR or low LPIPS proves that reconstructed buildings are reliable." (RGB pixel metrics are dominated by background landscape and are insensitive to building omission, boundary drift, or hallucination).
- *Forbidden Claim 4*: "Cross-correlation shifts demonstrate sub-pixel geodetic accuracy." (Cross-correlation measures only relative visual texture alignment, not geodetic truth).
