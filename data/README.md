# Data Directory Architecture

The `data/` directory organizes raw and processed geospatial imagery across the training, validation, and testing stages.

## Directory Layout

```
data/
├── training_data/                      # Training sets (Urban ~1.1M pairs, Full ~44M pairs)
│   ├── train_urban_set/               # Urban S2-NAIP training pairs
│   ├── train_full_set/                # Full nationwide S2-NAIP pairs
│   ├── osm_weights/                   # OpenStreetMap frequency weights & polygon masks
│   └── README.md                      # Detailed download and layout guide
├── validation_data/                    # Validation datasets and documentation
│   └── README.md                      # Guide to small_val_set, val_set, and pilot sets
├── test_data/                          # Global test sets (16x16 grids)
│   └── README.md                      # Guide for inference without ground-truth
├── small_val_set/                      # Extracted 256-tile urban validation subset
│   ├── naip/                          # 128x128 px NAIP 2.5m reference chips
│   └── sentinel2/                     # [T*32, 32, 3] Sentinel-2 time series stacks
├── small_val_set_tci_pilot/            # 5-tile pilot subset for rapid smoke testing
├── small_val_set_tci_five_tiles/       # 5-tile isolated benchmark set
└── super_resolution_small_val_set_2024-01-09.tar # Frozen validation archive
```

## Coordinate System & Format Specifications
1. **Web-Mercator Tiling**: Tiles follow the $2^{17} \times 2^{17}$ slippy map tile grid at Zoom 17. Each tile is indexed by `{x}_{y}` coordinates.
2. **Resolution & Upsampling Factor**:
   - Sentinel-2: $32 \times 32$ px at ~10 m/pixel ground sample distance (GSD).
   - NAIP (Ground Truth): $128 \times 128$ px at 2.5 m/pixel (downsampled $4\times$ from raw 0.6m NAIP).
   - Super-Resolution Factor: $4\times$ spatial resolution increase.
3. **Temporal Multi-Frame Stacking**:
   - Time series of $T$ revisit captures within a 3-month window are stacked vertically into a single PNG of dimensions $[32 \times T, 32, C]$.
   - Models slice this stack into $T$ separate $[C, 32, 32]$ tensors and fuse them across channels (e.g. $8 \times 3 = 24$ input channels for 8 revisits).

---

## Dataset Provenance & Global Test Grids

### 1. Sentinel-2 (ESA Copernicus)
- **Source**: European Space Agency (ESA) Copernicus Sentinel-2A & Sentinel-2B satellites (L1C Top-Of-Atmosphere reflectance).
- **Native Resolution**: $10\text{m}$ GSD across RGB (B04, B03, B02) and NIR (B08) bands.
- **Tiling**: Re-projected into Web-Mercator Zoom 17 ($9.555\text{m}$ nominal pitch), with 8 temporal observations per tile.

### 2. NAIP Aerial Imagery (USDA FSA)
- **Source**: United States Department of Agriculture (USDA) National Agriculture Imagery Program (NAIP) airborne surveys (2019–2020) across the contiguous US.
- **Native Resolution**: $0.6\text{m} - 1.0\text{m}$ GSD aerial orthophotography, downsampled $4\times$ to $2.5\text{m}$ GSD ($128 \times 128$ px) to provide the benchmark ground truth.

### 3. Global Test Grids: Kenya & Kota, India
- **Nakuru, Kenya (`figures/kenya_*.gif`)**: Sourced from raw Sentinel-2 L1C imagery over Nakuru County, Kenya (East Africa). Used as the project's flagship teaser to demonstrate zero-shot multi-frame super-resolution in regions outside the US without aerial NAIP coverage.
- **Kota, Rajasthan, India (`figures/kota_india.svg`)**: Sourced from Sentinel-2 L1C paired with high-resolution commercial Mapbox Satellite imagery (`ssr/options/experiments/india_finetune.yml`). Used to analyze out-of-domain geographic generalization limits and inductive bias breakdown when deploying US-trained priors on dense South Asian masonry architectures.


