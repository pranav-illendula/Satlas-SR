# EXP-013: Geospatial Metadata and Dataset Readiness Audit

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Python Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Audit Type**: Strict Read-Only Geospatial Alignment & Metadata Inventory  
**Target Research Direction**: "Uncertainty-Calibrated Satellite Image Super-Resolution for Reliable Building Mapping"  
**Audit Status**: COMPLETED  
**Date**: 2026-10-23  

---

## 1. Executive Summary

This audit assesses the local dataset and metadata readiness to support scientifically defensible research on **uncertainty-calibrated satellite image super-resolution for building mapping**. The investigation was conducted strictly read-only, without modifying any codebase, running inference, or downloading external data.

### Key Audit Findings:
1. **Tile Inventory Verified**: The local evaluation dataset in `..\data\small_val_set` contains exactly **256 Sentinel-2 folders** and **256 NAIP folders**, yielding **256 / 256 matching tile IDs** (`X_Y` format). There are 0 missing pairs, 0 duplicates, and 0 corrupted images.
2. **Geospatial Metadata Stripped**: All 2,816 local image files are standard `.png` images. Their PIL metadata dictionaries (`im.info`) are completely empty (`{}`). No GeoTIFF tags, ESRI world files (`.tfw`, `.pgw`), Shapefiles, GeoJSON boundaries, or XML sensor metadata exist locally.
3. **Temporal Asynchrony & Absence of Timestamps**: While NAIP flight dates can be partially recovered from USGS quadrangle names (spanning **2019-04-12 to 2019-12-15** across 239 unique quadrangles), **zero acquisition timestamps exist for the Sentinel-2 frames**. The 32 Sentinel-2 frames represent an unknown temporal composite.
4. **Tile Geometry**: Tile IDs correspond to **EPSG:3857 (Web Mercator) Slippy Map tiles at Zoom Level 17** ($z=17$). While bounding boxes can be analytically computed, sub-pixel orthorectification, sensor view angles, and local terrain distortions cannot be established without original Level-1C/Level-2A metadata.
5. **Building Reference Data Absent**: **Zero building footprints, segmentation masks, or boundary annotations exist locally**.

---

## 2. Dataset File and Modality Inventory

Direct inspection was conducted using [`inspect_data.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/inspect_data.py).

### 2.1. File Count and Modality Breakdown

| Modality / Target | Directory Path | File Count | Format | Channels & PIL Mode | Image Dimensions | Sub-frame Layout |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentinel-2 Multi-Spectral** | `..\data\small_val_set\sentinel2\<tile_id>\` | 2,304 | PNG | 1 (Mode `L`, 8-bit unsigned) | $32 \times 1024$ | 32 vertical frames of $32 \times 32$ |
| **Sentinel-2 True Color (TCI)** | `..\data\small_val_set\sentinel2\<tile_id>\tci.png` | 256 | PNG | 3 (Mode `RGB`, 8-bit unsigned) | $32 \times 1024$ | 32 vertical frames of $32 \times 32 \times 3$ |
| **NAIP Aerial Reference** | `..\data\small_val_set\naip\<tile_id>\rgb.png` | 256 | PNG | 3 (Mode `RGB`, 8-bit unsigned) | $128 \times 128$ | 1 static aerial image ($128 \times 128 \times 3$) |
| **Total** | | **2,816** | PNG | | | |

- **Sentinel-2 Bands Present** (9 single-channel bands per tile): `b01.png`, `b05.png`, `b06.png`, `b07.png`, `b08.png`, `b09.png`, `b10.png`, `b11.png`, `b12.png`. In addition, `tci.png` contains the pre-rendered True Color Image (Bands 4, 3, 2).
- **Sub-frame Stacking**: As documented in [`inspect_data.py:L66-L77`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/inspect_data.py#L66-L77), each Sentinel-2 PNG has width $W=32$ and height $H=1024$, which stores 32 temporal observations ($1024 / 32 = 32$) stacked vertically.
- **NAIP Reference**: Each tile folder contains exactly one file: `rgb.png` ($128 \times 128 \times 3$).

### 2.2. Tile Matching and Integrity Verification

- Total Sentinel-2 tile folders: **256**
- Total NAIP tile folders: **256**
- Matching tile ID intersection: **256 / 256 (100.0%)**
- S2-only tile IDs: **0**
- NAIP-only tile IDs: **0**
- Corrupted or unreadable image files: **0** (verified via PIL `image.verify()` and `image.load()` on all 2,816 files).

---

## 3. Comprehensive Metadata Classification

Every candidate geospatial and sensor metadata field required for rigorous remote sensing evaluation has been inspected and classified into one of three strict evidentiary categories:
1. **Verified and Available**: Directly present and verified from local files.
2. **Partially Available or Uncertain**: Recoverable by inference or proxy, but lacking authoritative sensor/geodetic verification.
3. **Not Found in Inspected Local Files**: Completely missing from local storage.

| Metadata Field | Classification Status | Evidence / Source in Local Files | Limitations & Impact on Research |
| :--- | :---: | :--- | :--- |
| **Tile Identifier (`X_Y`)** | **Verified and Available** | Folder names in `sentinel2/` and `naip/` | Establishes pair correspondence, but does NOT establish geodetic alignment. |
| **Map Projection / CRS** | **Verified and Available** | Web Mercator Slippy Map format (EPSG:3857) | Standard web tiling projection; high latitude conformal distortion. |
| **Slippy Map Zoom Level ($z=17$)** | **Verified and Available** | Inferred from $X, Y$ coordinate ranges ($X \in [18000, 35000]$, $Y \in [45000, 55000]$) | Grid resolution is fixed at $2^{17} = 131,072$ tiles along equator. |
| **Tile Bounding Box (Corners)** | **Verified and Available** | Calculated via Slippy Map equations: $\text{lon} = \frac{X}{2^z}\cdot 360 - 180$; $\text{lat} = \arctan(\sinh(\pi(1 - \frac{2Y}{2^z})))\cdot \frac{180}{\pi}$ | Provides nominal bounding box corners; does not capture sub-pixel terrain displacement. |
| **Ground Sampling Distance (GSD)** | **Partially Available / Uncertain** | Web Mercator nominal pixel size: $\text{GSD} = \frac{40075016.686 \cdot \cos(\text{lat})}{256 \cdot 2^{17}}$ | Nominal NAIP pixel GSD $\approx 2.39\text{ m/px}$, S2 $\approx 9.55\text{ m/px}$. Actual airborne NAIP GSD was natively $0.6\text{ m}$ or $1.0\text{ m}$ before downsampling/resampling into Slippy tiles. |
| **NAIP Acquisition Date** | **Partially Available / Uncertain** | Extracted from USGS quadrangle names (e.g., `m_4612249_nw_10_060_20190527`) | Dates span **2019-04-12 to 2019-12-15** across 239 quads. Exact time of day, sun angle, and flight line geometry are missing. |
| **Sentinel-2 Acquisition Dates** | **Not Found Locally** | Inspected all 2,560 S2 PNGs and directory paths | **Zero timestamps** exist for the 32 Sentinel-2 frames. Unknown whether frames span months or years relative to NAIP. |
| **Sentinel-2 Solar Geometry** | **Not Found Locally** | Not present in any local file | Solar zenith, azimuth, and illumination angles are unknown. |
| **Sentinel-2 Sensor View Geometry** | **Not Found Locally** | Not present in any local file | Off-nadir viewing angle and parallax cannot be modeled. |
| **Cloud / Cloud-Shadow Masks** | **Not Found Locally** | No QA60 bands, scene classification layer (SCL), or cloud probability masks | Quality screening relies solely on heuristic pixel brightness proxies (e.g., EXP-011 Strategy S6). |
| **Affine Transformation Matrix** | **Not Found Locally** | PNG files lack World Files (`.tfw`, `.pgw`) or GeoTIFF tags | Sub-pixel raster-to-coordinate mapping is impossible without external lookup. |
| **Building Footprints / Masks** | **Not Found Locally** | Searched entire workspace; 0 vector or raster masks exist | **Evaluation of building boundaries, IoU, or structural fidelity is impossible with local data alone.** |

---

## 4. Analysis of Geospatial Information Loss During Dataset Extraction

The Satlas-SR data extraction pipeline transformed original satellite Level-1C/Level-2A SAFE packages and USGS NAIP GeoTIFFs into Web Mercator PNG tiles. This process introduced several irreversible information losses in the local dataset:

1. **Loss of Geodetic Metadata**:
   - Original GeoTIFFs contain EPSG codes, UTM zone definitions, exact corner coordinates, pixel scales, and rational polynomial coefficients (RPCs) or orthorectification grids.
   - Converting to PNG stripped all header metadata, leaving integer arrays (`uint8`, $[0, 255]$).
2. **Loss of Radiometric Calibration**:
   - Original Sentinel-2 Level-2A imagery contains Bottom-of-Atmosphere (BOA) surface reflectance in units of 1/10,000 (reflectance values $0.0 - 1.0$).
   - The PNGs are non-linearly tone-mapped or stretched to 8-bit unsigned integers, preventing physical radiance/reflectance comparisons.
3. **Loss of Temporal Metadata**:
   - Sentinel-2 multi-frame super-resolution relies on multi-temporal observations. Without timestamps, the temporal baseline (revisit interval, seasonal distribution, drought/snow variations) is completely uncharacterized.
4. **Resampling and Interpolation Artifacts**:
   - Web Mercator tiling requires resampling native UTM Sentinel-2 pixels ($10\text{ m}$) and native NAIP pixels ($0.6\text{ m}$ or $1.0\text{ m}$) into zoom-17 pixel grids ($\sim 9.55\text{ m}$ and $\sim 2.39\text{ m}$).
   - This resampling introduces bilinear/bicubic smoothing and subtle sub-pixel coordinate shifts.

---

## 5. Traceability and Evidentiary References

- **Dataset Inspection Script**: [`inspect_data.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/inspect_data.py)
- **Baseline Manifest**: [`results/satlas_sr_baseline_freeze_v1/baseline_manifest.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_baseline_freeze_v1/baseline_manifest.json)
- **Frame Selection Audit**: [`results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/full_strategy_summary.json`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_011_frame_selection_audit/full_256_tiles/full_strategy_summary.json)
- **Registration Audit**: [`results/satlas_sr_spatial_audit/registration_validation_report.md`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_spatial_audit/registration_validation_report.md)
