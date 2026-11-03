# Validation Data Directory

This directory references and manages validation datasets used for qualitative checks and quantitative evaluation.

## Available Datasets

### 1. Small Validation Set (`data/small_val_set`)
- **Scale**: 256 paired tiles from urban areas.
- **Purpose**: Fast regression testing, qualitative visual comparisons, and benchmark verification.
- **Archive**: `data/super_resolution_small_val_set_2024-01-09.tar` (34.9 MB).
- **Download**: [super_resolution_small_val_set_2024-01-09.tar](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/super_resolution_small_val_set_2024-01-09.tar)

### 2. Pilot Subset (`data/small_val_set_tci_pilot`)
- **Scale**: 5 representative tiles:
  - `20771_46525`
  - `20890_46801`
  - `20926_46045`
  - `21068_45692`
  - `21856_49804`
- **Purpose**: Instant smoke testing and verification of inference / metric calculation within seconds.

### 3. Full Validation Set (`val_set`)
- **Scale**: 8,192 paired tiles across diverse US biomes.
- **Download**: [super_resolution_val_set_2024-01-09.tar](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/super_resolution_val_set_2024-01-09.tar)

