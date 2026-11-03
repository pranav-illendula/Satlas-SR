# Inference Pipelines

This directory contains standalone execution scripts for running super-resolution inference on raw Sentinel-2 satellite imagery.

## Scripts

| Script | Purpose | Key Flags | Expected Output |
| :--- | :--- | :--- | :--- |
| `run_pilot_inference.py` | Runs inference across the 5 pilot urban tiles | `-opt ssr/options/infer_satlas_five_tiles.yml` | `results/satlas_sr_pilot_five_tiles/` |
| `run_full_validation_inference.py` | Runs inference across all 256 validation tiles | `-opt ssr/options/infer_satlas_full_validation.yml` | `results/satlas_sr_full_validation/` |

## Usage Examples

### 1. Run 5-Tile Pilot Inference
```bash
python inference/run_pilot_inference.py -opt ssr/options/infer_satlas_five_tiles.yml
```

### 2. Run Full Validation Set Inference
```bash
python inference/run_full_validation_inference.py -opt ssr/options/infer_satlas_full_validation.yml
```

### 3. Native Model Inference (via SSR engine)
```bash
# Single tile
python -m ssr.infer -opt ssr/options/infer_example.yml

# 16x16 grid mosaic stitching
python -m ssr.infer_grid -opt ssr/options/infer_grid_example.yml
```

