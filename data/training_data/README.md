# Training Data Directory

This directory is designated for storing the training datasets used to train the Satlas Super-Resolution (Satlas-SR) models.

## Datasets Overview

### 1. Urban Training Set (`train_urban_set` / S2-NAIP)
- **Scale**: ~1.1 million image pairs.
- **Coverage**: Locations within a 5 km radius of US cities with population $\ge 50,\!000$.
- **Spectral Bands**: 12 Sentinel-2 bands (TCI RGB, B01, B05, B06, B07, B08, B09, B10, B11, B12).
- **Download Links**:
  - [Part 1 (7z.001)](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/train_urban_set.7z.001)
  - [Part 2 (7z.002)](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/train_urban_set.7z.002)
  - [Part 3 (7z.003)](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/train_urban_set.7z.003)
  - [Part 4 (7z.004)](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/train_urban_set.7z.004)
- **HuggingFace Hub**: Available at [allenai/satlas-super-resolution](https://huggingface.co/datasets/allenai/satlas-super-resolution/tree/main).

### 2. Full Training Set (`train_full_set`)
- **Scale**: ~44 million image pairs.
- **Coverage**: All locations where NAIP imagery was captured between 2019-2020 across the contiguous USA.
- **Download Link**: [super_resolution_train-full-set_2023-12-01.tar](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_train-full-set_2023-12-01.tar)

### 3. OpenStreetMap Weightings & Polygon Masks (`osm_weights`)
- **Tile Weights**: [super_resolution_train_tile_weights.tar](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/super_resolution_train_tile_weights.tar)
- **OSM Chips to Masks**: [osm_chips_to_masks.json](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/osm_chips_to_masks.json)

---

## Expected Directory Layout

When extracted, the training dataset must follow this directory structure:

```
data/training_data/train_urban_set/
├── naip/
│   └── {image_uuid}/
│       └── {tile_id}/
│           └── rgb.png               # 128x128 px, RGB, 25% NAIP resolution (2.5m/px)
└── sentinel2/
    └── {tile_id}/
        ├── tci.png                   # Stitched vertical stack: [T * 32, 32, 3] px
        ├── b01.png                   # Band 01 stack (60m native)
        ├── b05.png                   # Band 05 stack (20m native)
        └── ...                       # Other Sentinel-2 multispectral bands
```

- `{tile_id}`: Standard Web-Mercator coordinate formatted as `{x}_{y}` in a $2^{17} \times 2^{17}$ grid (e.g. `36226_55652`).
- `{image_uuid}`: Unique identifier including timestamp of the NAIP flight survey.
- Vertical Stack: Each Sentinel-2 stack PNG encodes $T$ temporal frames ($T \in [1, 16]$) stacked vertically as a $[32 \times T, 32, 3]$ image.

