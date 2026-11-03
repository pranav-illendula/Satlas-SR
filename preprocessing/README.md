# Preprocessing & Data Ingestion Pipeline

The `preprocessing/` package houses scripts and modules for preparing, validating, and auditing Sentinel-2 and NAIP datasets prior to model training and evaluation.

## Scripts Overview

| Script | Purpose | Key Inputs | Primary Output |
| :--- | :--- | :--- | :--- |
| `inspect_data.py` | Fast inspection of dataset structure, dimensions, modes, stack distributions | Dataset root path | Console summary & error count |
| `audit_dataset.py` | Rigorous health audit checking tile pairing, missing bands, frame counts, black frames | Dataset root path | Health status & JSON audit log |
| `audit_spatial_registration.py` | Measures relative sub-pixel translational shift between S2 and NAIP | Reference and moving images | 2D phase shift $(\Delta x, \Delta y)$ & NCC |
| `reproject_sentinel2.py` | Reprojects raw Sentinel-2 TCI `.jp2` products to Web-Mercator EPSG:3857 | Raw ESA `.jp2` file | 9.555m reprojected GeoTIFF |

---

## Usage Examples

### 1. Inspect Dataset
Verify that all Sentinel-2 stacks are intact and match their corresponding NAIP ground truth chips:
```bash
python preprocessing/inspect_data.py data/small_val_set --samples 2
```

### 2. Comprehensive Dataset Audit
Audit every single tile for corruption, zero-byte errors, and frame validity:
```bash
python preprocessing/audit_dataset.py data/small_val_set --output-json results/dataset_audit.json
```

### 3. Spatial Registration Analysis
Measure alignment between a low-resolution input preview and high-resolution NAIP reference:
```bash
python preprocessing/audit_spatial_registration.py \
  --ref data/small_val_set/naip/m_3211716_ne_11_060_20190317_20190412/22917_52849/rgb.png \
  --moving data/small_val_set/sentinel2/22917_52849/tci.png
```

### 4. Raw Sentinel-2 Reprojection
Convert raw ESA Sentinel-2 L1C granules to the 9.555m Web-Mercator projection required by Satlas-SR:
```bash
python preprocessing/reproject_sentinel2.py \
  --input-jp2 /path/to/TCI.jp2 \
  --output-dir data/processed_tiles/ \
  --resolution 9.555
```

