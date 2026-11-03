# Satlas-SR provenance manifest request

**Prepared:** 2026-10-19  
**Scope:** Identify the smallest authoritative metadata deliverable needed to decide whether the existing 256-tile baseline predictions can be compared to their NAIP targets using validated PSNR/SSIM. This report was prepared from local files and official, publicly accessible documentation. No author was contacted, no data was downloaded, and no inference or evaluation was run.

## Evidence gap

The released PNGs, local inference manifest, and repository code establish tile-key pairing and record which S2 stack positions were selected. They do not map each stack position to an S2 scene/product or acquisition time, nor do they establish per-image source/output transforms, crop windows, pixel-grid origin, or applied resampling/alignment. The local README gives an illustrative S2 reprojection recipe, not evidence that the recipe was applied to these exact files. The evaluation report explicitly labels registration unverified; its PSNR/SSIM are exploratory raw-array comparisons.

The paper says geospatial and temporal information is provided for all imagery and describes an S2 temporal window within two months of NAIP capture. The official repository README instead describes a three-month window. The accessible official repository/dataset listings reviewed here did not expose a small-validation per-image provenance manifest or data-generation scripts. The metadata may be distributed separately; its exact location and schema remain unresolved. [Paper, Section 4.1](https://arxiv.org/html/2311.18082v1#S4.SS1) · [Official Satlas-SR repository](https://github.com/allenai/satlas-super-resolution/tree/experiments) · [Official dataset listing](https://huggingface.co/datasets/allenai/satlas-super-resolution)

## Required fields and findings

| Required field | Why it matters | Found in reviewed sources? |
|---|---|---|
| Satlas-SR tile ID and exact S2 PNG identity | Joins provenance to the existing input tile without ambiguity | Tile IDs and paths exist locally; no source manifest key beyond the tile path was found. |
| S2 stack frame index (zero-based) | Maps the saved PNG stack position and inference-selected indices to a source observation | Stack indices are recorded for inference; no authoritative source-scene mapping by index was found. |
| S2 scene/product ID and acquisition timestamp with timezone | Verifies which observation each selected frame represents and its temporal distance from NAIP | No per-frame ID or date. README warns stack order need not be chronological. |
| NAIP image UUID/source product ID and authoritative acquisition datetime | Establishes the reference acquisition identity and temporal comparison date | UUID/date-like token appears in the folder convention; local source product catalog and timestamp semantics were not found. |
| NAIP source footprint and the exact 128x128 chip footprint | Confirms the ground reference covers the intended tile and identifies any clipping/offset | README gives nominal zoom-17 Web-Mercator tile organization; no per-file footprint or chip bounds were found. |
| For each source and derived image: CRS and affine transform, or equivalent precise bounds plus pixel size/grid origin | Establishes the mapping from pixel coordinates to ground coordinates for independent registration checks | No per-image CRS, affine, origin, or bounds in local PNGs, manifests, repository, or accessible dataset-card structure reviewed. |
| Source-to-chip crop window and output chip dimensions | Connects each delivered PNG to its source pixels and allows the crop transform to be reconstructed | No per-image crop window/source pixel window found. README documents 128x128 NAIP and 32x32 S2 frame shapes only. |
| Resampling method/parameters and processing software/version for S2 and NAIP | Detects shifts or pixel-phase changes introduced while reprojecting/downsampling | README’s EPSG:3857, 9.555 m, bilinear example is generic; application to these exact files is unverified. NAIP is described as quarter-resolution, but the kernel/phase is unspecified. |
| Pairing rule, temporal inclusion window, and registration/alignment method or explicit “none” | Makes the pair selection and any correction reproducible; resolves the paper/README window discrepancy | Broad, conflicting two-/three-month statements exist. Exact per-pair method and any alignment correction are not documented in reviewed sources. |
| Valid-data/cloud/NoData mask reference for each S2 frame | Identifies pixels that should not contribute to comparison and supports cloud screening | Paper says cloudy images were retained; no frame-specific mask or status found. |

**Verified local context:** `results/satlas_sr_full_validation/inference_manifest.csv` records the selected stack indices, input/output paths, and run identifiers; `run_summary.md` records the checkpoint SHA-256, `params_ema`, config, seed, and timing. These are useful provenance for the model run, but they do not identify the upstream S2 scenes or image geotransforms. `summary.json` and `reports/satlas_sr_full_validation_evaluation.md` label spatial registration `UNVERIFIED` and mark raw-array metrics exploratory. The extracted archive inventory contains PNGs only. The local loader associates NAIP and S2 by tile key (`ssr/data/s2-naip_dataset.py`); inference utility reshapes and randomly samples frames (`ssr/utils/infer_utils.py`).

**Official-source scope:** The paper’s metadata statement is evidence that the authors intended geospatial/temporal metadata to be available, not proof that the small-val archive contains it. The official README documents the small-val tar and broad layout; the public Hugging Face listing reviewed exposes the dataset split/image-directory structure but not a dedicated small-val manifest. The paper and accessible listings do not expose the required per-file transforms or frame-to-scene table. The supplementary PDF is linked from the repository README, but was not accessible via the no-auth research route used for this task; its contents are therefore unresolved. No information from the separate SatlasPretrain product is assumed to map this Satlas-SR package.

## Minimal author-provided manifest

The most compact practical deliverable is one JSON object per tile (JSONL is suitable for the 256-tile subset), with an array containing **every** S2 stack frame. This is preferable to a full source-data dump. Source asset IDs/URIs can substitute for repeating source georeferencing fields only if the referenced asset metadata is publicly retrievable and version-stable.

```json
{
  "schema_version": "1.0",
  "tile_id": "20771_46525",
  "pairing": {
    "rule": "<spatial-overlap and temporal inclusion rule used>",
    "temporal_window": "<exact bound and whether inclusive>",
    "registration_method": "<documented method, or none>",
    "correction_applied": "<none, or method and parameters>"
  },
  "naip": {
    "png_path": "naip/<image_uuid>/<tile_id>/rgb.png",
    "image_id": "<NAIP image UUID and source product/scene ID>",
    "acquired_at": "<ISO-8601 timestamp with timezone or documented date precision>",
    "source_asset": "<stable URI or catalog/product ID>",
    "source_crs": "<authority code or WKT>",
    "source_transform": ["<six affine coefficients, or equivalent>"],
    "source_footprint": "<bounds or GeoJSON geometry in source_crs>",
    "chip_crs": "<authority code or WKT>",
    "chip_transform": ["<six affine coefficients, or equivalent>"],
    "chip_bounds": ["<xmin>", "<ymin>", "<xmax>", "<ymax>"],
    "crop_window_pixels": ["<column_off>", "<row_off>", "<width>", "<height>"],
    "chip_size_pixels": [128, 128],
    "pixel_size": ["<x_size>", "<y_size>"],
    "resampling": "<kernel and parameters>",
    "processing_software": "<name and version>"
  },
  "sentinel2_frames": [
    {
      "stack_index_zero_based": 0,
      "png_path": "sentinel2/<tile_id>/tci.png",
      "scene_id": "<Sentinel-2 scene/product ID>",
      "acquired_at": "<ISO-8601 timestamp with timezone>",
      "source_asset": "<stable URI or catalog/product ID>",
      "source_crs": "<authority code or WKT>",
      "source_transform": ["<six affine coefficients, or equivalent>"],
      "source_footprint": "<bounds or GeoJSON geometry in source_crs>",
      "chip_crs": "<authority code or WKT>",
      "chip_transform": ["<six affine coefficients, or equivalent>"],
      "chip_bounds": ["<xmin>", "<ymin>", "<xmax>", "<ymax>"],
      "crop_window_pixels": ["<column_off>", "<row_off>", "<width>", "<height>"],
      "chip_size_pixels": [32, 32],
      "pixel_size": ["<x_size>", "<y_size>"],
      "resampling": "<kernel and parameters>",
      "valid_mask_or_cloud_mask": "<asset/path, or explicit unavailable>",
      "processing_software": "<name and version>"
    }
  ]
}
```

The authors need not regenerate imagery if these details already exist in source metadata/logs. If a field is not recorded, it should be reported as unavailable rather than reconstructed by matching the ESRGAN prediction. If transforms are exactly the standard zoom-17 cell grid, the manifest can state the precise tile-coordinate convention and give the resulting CRS, bounds, and affine once per relevant grid definition; it must still confirm that both PNG chips were actually created on that grid.

## Draft email (not sent)

**Subject:** Request for small-val S2/NAIP provenance manifest and preprocessing details

Hello Satlas-SR team,

I am auditing the released eight-input ESRGAN baseline on the 256-image `small_val_set`. The paper says geospatial and temporal metadata are provided for S2-NAIP, but the small-val PNG archive and public repository files I reviewed do not include a per-frame scene/date mapping or per-chip transforms. The README describes a three-month overlap window, while paper Section 4.1 describes two months.

Could you point me to the smallest existing manifest or metadata file covering this small validation subset? For each tile, we need the NAIP image/source ID, acquisition date and footprint; for every S2 stack frame, its zero-based stack index, Sentinel-2 scene/product ID and acquisition time; and the CRS, affine/bounds, crop window, pixel size, resampling steps, and any registration correction used to create the delivered PNGs. Please also confirm the temporal pairing rule and whether clouds/NoData masks are available. A CSV/JSON table or the exact preprocessing scripts plus a stable source metadata reference would be sufficient; we do not need the full imagery archive.

If any fields were not recorded, please let us know which are unavailable. We will keep the current image-pair metrics exploratory unless spatial registration and temporal comparability can be independently established.

Thank you,  
<Name>

## Recommendation

**NO-GO for validated PSNR/SSIM at present.** Existing exploratory metrics must not be presented as validated performance. The minimum next step is to send the draft request through the project’s published contact channel and obtain either the compact tile/frame manifest described above or a precise pointer to its official location. No contact has been made as part of this investigation. After receipt, independently verify the recorded transforms, overlap, frame dates, cloud/valid masks, and processing history before computing any validated pixel metrics.

## Files inspected and created

Inspected local files:

- `satlas-super-resolution/reports/satlas_sr_provenance_investigation.md`
- `satlas-super-resolution/README.md`
- `satlas-super-resolution/ssr/data/s2-naip_dataset.py`
- `satlas-super-resolution/ssr/utils/infer_utils.py`
- `satlas-super-resolution/ssr/options/infer_satlas_full_validation.yml`
- `satlas-super-resolution/results/satlas_sr_full_validation/run_summary.md`
- `satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv`
- `satlas-super-resolution/results/satlas_sr_full_validation/summary.json`
- `satlas-super-resolution/reports/satlas_sr_full_validation_evaluation.md`
- Local repository file inventory for metadata/provenance/source-processing artifacts.

Official public sources inspected:

- [Paper, Section 4.1](https://arxiv.org/html/2311.18082v1#S4.SS1)
- [Satlas-SR official `experiments` branch](https://github.com/allenai/satlas-super-resolution/tree/experiments)
- [Satlas-SR Hugging Face dataset](https://huggingface.co/datasets/allenai/satlas-super-resolution)

Created:

- `satlas-super-resolution/reports/satlas_sr_provenance_request.md`
