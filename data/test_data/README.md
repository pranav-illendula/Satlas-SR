# Test Data Directory

This directory is designated for evaluating inference on unannotated Sentinel-2 grids without paired ground-truth imagery.

## Overview
- **Paper Test Set**: Contains eight $16 \times 16$ grids of Sentinel-2 tiles covering global locations (Dry Tortugas National Park, Bolivia, France, South Africa, Japan).
- **Download Link**: [super_resolution_test_set_2024-01-09.tar](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/super_resolution_test_set_2024-01-09.tar)
- **Usage with Grid Inference**:
  Run `python -m satlas-super-resolution.ssr.infer_grid -opt ssr/options/infer_grid_example.yml` to infer and stitch contiguous $16 \times 16$ tile mosaics.

