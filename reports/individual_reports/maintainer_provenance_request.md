# Draft request for Satlas-SR validation provenance

**Subject:** Provenance and pixel-grid details for the Satlas-SR `small_val_set`

Hello Satlas-SR maintainers,

I am auditing the released `small_val_set` for pixel-level evaluation of super-resolution outputs. The archive README describes 256 urban image pairs, but the released PNGs do not include geospatial sidecars or source-scene identifiers. Could you share the data-generation provenance for this validation subset, or point me to an existing manifest or source pipeline?

The most useful fields would be:

1. For each NAIP chip: its source scene/product identifier, acquisition date (and time if available), footprint, CRS, source affine transform, and exact crop/window used to create the 128x128 chip.
2. For every Sentinel-2 frame in each stack: scene/product identifier, acquisition date/time, original CRS and affine transform, and the mapping from stack position to scene.
3. The common target grid used for each pair: tile bounds, pixel size, pixel origin, crop windows, and any alignment correction.
4. The processing steps actually applied to these files, including reprojection parameters, resampling methods, and any clipping or resizing. In particular, was the README's EPSG:3857 / 9.555 m TCI example applied to these exact validation files?
5. Whether per-observation registration residuals or valid-overlap checks were computed, and any acceptance criteria used.

The current archive and local code establish shared tile IDs and intended pairing, but not pixel-level registration. Dataset-level documentation says S2 observations overlap their NAIP image within three months; per-frame dates and transforms would let us evaluate registration and temporal comparability without estimating a flexible image warp.

The five tiles already audited locally are `20771_46525`, `20926_46045`, `20890_46801`, `21068_45692`, and `21856_49804`. If it is easier, provenance for these five would be a useful first step. We can provide the corresponding NAIP folder identifiers and archive details on request.

Thank you.

---

This is a draft only. It has not been sent. Upstream public documentation checked on 2026-10-19 includes the [Satlas-SR README](https://github.com/allenai/satlas-super-resolution/blob/main/README.md#s2-naip-dataset-structure) and the [Hugging Face dataset mirror](https://huggingface.co/datasets/allenai/satlas-super-resolution/tree/refs%2Fpr%2F3/ssr-data/val_set).
