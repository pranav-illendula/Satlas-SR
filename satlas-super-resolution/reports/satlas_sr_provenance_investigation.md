# Satlas-SR authoritative provenance investigation

**Investigation date:** 2026-10-19  
**Scope:** Read-only review of the local Satlas-SR repository and `small_val_set`, plus primary upstream documentation, paper, repository, and dataset listings. No images, dataset files, inference outputs, checkpoint, configuration, environment, or Git history were changed. No data was downloaded and no inference/evaluation was run.

## Executive summary

The authors document an intended pairing: NAIP image chips and Sentinel-2 time series share a Web-Mercator zoom-17 tile identifier, with temporal overlap described as within three months by the repository README. The paper describes S2 images within two months of NAIP acquisition and says geospatial and temporal metadata are provided for all imagery. These are dataset-level statements.

The extracted `small_val_set` package contains 2,816 PNG files and no sidecar metadata; the local repository contains no dataset generation pipeline or source-scene manifest. The PNGs and inference code do not record individual S2 scene IDs/dates, georeferencing transforms, exact crop windows, or source resampling history. The official Hugging Face dataset listing examined exposes split directories and image folders; no per-observation provenance sidecar was identified in its top-level validation structure. The separately linked supplementary PDF could not be retrieved by the read-only browser research tool (HTTP 401), and is not present locally, so no supplementary claims beyond the paper/repository references are made here.

**Decision: NO-GO for validated pixel-level evaluation using the current PNG package.** Matching tile IDs permit a nominal grid-cell association, and conventional Web-Mercator tile bounds can be derived from a tile coordinate under a specified tiling convention. They do not establish the actual per-file pixel-to-world transform, matching crop origin, resampling phase, or temporal compatibility. The existing PSNR/SSIM remain exploratory same-index array comparisons. Independent validation becomes feasible if the maintainers provide authoritative per-image source records and processing provenance.

## 1. Verified local evidence

### Author-documented statements

- `README.md` describes the dataset as NAIP/Sentinel-2 pairs, with an S2 time series that overlaps the NAIP image spatially and temporally “within 3 months.” It says the imagery is from 2019–2020 and uses the same Web-Mercator tile system as SatlasPretrain (`README.md`, dataset structure section).
- The README gives NAIP paths as `naip/{image_uuid}/{tile}/rgb.png`; it says the UUID carries the capture timestamp and the tile is in a `2^17 x 2^17` Web-Mercator grid. It describes NAIP PNGs as 128x128 RGB at one quarter the original NAIP resolution.
- The README says Sentinel-2 L1C TCI is used for RGB input, TCI is an 8-bit ESA-normalized product, and each NAIP image has a stack of 32x32 S2 frames in `sentinel2/{tile}/tci.png`. It explicitly warns that frame order need not be chronological.
- The README gives a *general raw-TCI processing example* using EPSG:3857, 9.555 m output resolution, and bilinear resampling. It does not tie that example to the source scenes or exact pixels in this validation archive, and gives no NAIP chip transform/crop recipe.
- The local `small_val_set` archive is 34,918,400 bytes. The extracted inventory has 2,816 files, all `.png`; the existing inventory reports the TAR has PNG members and directory entries only. No JSON/CSV/XML/world files, raster source products, or geospatial sidecars are present in the package.

### Code-established behavior (inference from implementation)

- `ssr/data/s2-naip_dataset.py` enumerates NAIP PNGs, takes the parent directory as the tile key, and constructs the S2 path from that same key (`:104-130`). This establishes tile-key association in the loader, not a coordinate-based registration check. The loader does not inspect transforms or compare spatial footprints.
- The loader reshapes S2 PNG stacks into 32x32 frames and samples usable frames (`:188-226`). Its optional random crop and resizing path is conditional on `rand_crop` (`:232-240`); this is not an authoritative record of how archived chips were originally created.
- `ssr/utils/infer_utils.py` reshapes the stacked PNG, samples frames via Python `random.sample`, and scales RGB values by 255. The selected frame indices are stack positions; without a frame manifest, they cannot be mapped to acquisition dates.
- `ssr/infer.py` reads PNG files and calls the formatter; it does not reproject, georeference, or align NAIP and Sentinel-2. The validation/inference configuration specifies model and paths, not per-image geospatial metadata.
- A repository file search found no included dataset-creation script, source-scene map, S2 frame manifest, NAIP source metadata table, or per-chip transform file. Local audit artifacts (`reports/registration_provenance_investigation.md`, `reports/satlas_sr_full_validation_evaluation.md`, and `results/satlas_sr_full_validation/summary.json`) likewise record that the PNG metadata is insufficient. The evaluation summary marks spatial registration `UNVERIFIED`.

### What local filenames establish

NAIP folder names end in date-like tokens (for example, `20191010`), and the README says the UUID contains a capture timestamp. This is useful NAIP-side naming evidence, but the extracted package has no source catalog confirming timestamp precision, original acquisition product, scene footprint, or chip transform. Sentinel-2 stack frames have no per-frame date or scene ID. No temporal ordering may be inferred from their vertical stack order.

The tile coordinate pair and stated Web-Mercator grid permit a *nominal* cell footprint to be calculated if the coordinate axis convention and zoom-17 tile scheme are fixed. The README does not supply file-level affine transforms or prove that each PNG was cropped/resampled to that exact grid cell origin. A nominal tile bbox is not a substitute for a raster transform.

## 2. Authoritative upstream evidence

### Paper

The paper’s Section 4.1 states that the authors source NAIP from 2019–2020, select spatially overlapping Sentinel-2 imagery within two months of each NAIP capture timestamp, retain cloudy imagery, and provide geospatial and temporal information for all imagery. It describes Web-Mercator zoom-17 tiles, 512x512 full-resolution NAIP tiles, 32x32 S2 inputs, and NAIP downsampled by four for a 128x128 target. This is evidence that such metadata exists or was intended to be available with the dataset; the paper text itself does not provide the small validation subset’s per-frame IDs, dates, transforms, or crop windows. [Paper, Section 4.1](https://arxiv.org/html/2311.18082v1#S4.SS1)

The paper’s two-month temporal window differs from the repository README’s three-month description. The paper also says clouds are not removed, so cloud presence and land-cover/building changes can affect comparisons even where the documented time bound is met. [Paper, Sections 3 and 4.1](https://arxiv.org/html/2311.18082v1)

### Official Satlas-SR repository

The official [`experiments` branch README](https://github.com/allenai/satlas-super-resolution/tree/experiments) publishes the small validation archive as `super_resolution_small_val_set_2024-01-09.tar` and describes its image-folder structure and illustrative raw-TCI reprojection. Its branch landing page lists code/configuration directories and README, not a data-generation pipeline or validation sidecar. The local `experiments`-style checkout was also searched and had no such source scripts or metadata.

The [official Hugging Face dataset card](https://huggingface.co/datasets/allenai/satlas-super-resolution) points to image paths in `ssr-data/{train_urban_set,test_set,val_set}`. Its repository root reports 974 GB and the validation set is arranged under `val_set/naip` and `val_set/sentinel2`; the published top-level validation structure does not identify a metadata manifest. The card does not expose a `small_val_set` metadata sidecar. The larger repository is not a reasonable target to download for this investigation, and its directory listing alone cannot prove that every file deeper in the entire corpus lacks metadata.

### Supplementary material

The official README links the supplementary PDF at `https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/Zooming_Out_On_Zooming_In_Supplementary.pdf`. The browser research tool returned HTTP 401 for that URL, and the supplement is not present in the local repository. I therefore could not verify whether it documents a schema, external metadata location, or per-image processing details. This is an unresolved source, not evidence that no such information exists.

### Similar SatlasPretrain metadata is a separate product

The [SatlasPretrain documentation](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) describes metadata files including `image_times.json` and tile-coordinate conversion utilities for SatlasPretrain. That is a separate dataset/product. No source establishes that its metadata maps the Satlas-SR validation PNG stack positions or proves the crop transforms used for this archive; those records must not be transferred to this dataset by name or shared tile convention alone.

## 3. Metadata availability

| Required record | Available in local package/repository? | Finding |
|---|---|---|
| Tile ID to nominal geographic bounds | Partially derivable | The README documents zoom-17 Web-Mercator grid IDs. A nominal cell bbox can be calculated under the standard tiling convention, but exact axis convention and per-file affine are not recorded. |
| NAIP source scene ID/date and footprint | Partial | NAIP folder UUID has a date-like capture token by documented convention. Source scene catalog, exact timestamp semantics, original footprint, and source-to-chip window/transform are absent. |
| S2 stack index to source scene ID/date | No | The PNG stack gives frame positions only; no per-frame S2 identifier, acquisition time, or chronological mapping was found. |
| Per-image CRS and affine transform | No | PNG archive and local manifests contain no georeferencing sidecars or raster metadata. |
| Exact NAIP/S2 crop bounds and pixel origin | No | No per-chip crop window, transform, grid origin, or registration adjustment record was found. |
| Applied S2 and NAIP resampling history | Partial/generic only | README illustrates S2 EPSG:3857 at 9.555 m with bilinear resampling, but does not verify those parameters for these exact chips. NAIP downsampling is described as 25% resolution / factor four, but kernel and origin are unspecified. |
| Cloud/valid-pixel mask per S2 frame | No | Paper says cloudy images were retained; no per-frame cloud mask/status is in the local package. |
| Source data-generation pipeline | No | Not in the local repository; not listed in the public code branch files inspected. |

The paper’s assertion that metadata is provided and the absence of such records in this archive are not contradictory: the records may have been published separately or distributed with another data artifact. Their exact source, schema, availability, and mapping to this `small_val_set` were not established here.

## 4. Registration-validation feasibility and plan

### Current feasibility: insufficient metadata

The present PNGs permit visual review and nominal tile-key matching, but not an independent transform-based validation. Equal dimensions, matching tile IDs, and shared intended tiling do not establish pixel-level registration or temporal comparability. No coordinates should be inferred from imagery, and no similarity-maximizing warp should be fitted to the ESRGAN output.

### Records needed to proceed

Request from the maintainers or recover from the original data-generation package:

1. **Pair/index manifest:** exact small-val tile ID, NAIP UUID/source product ID, and the ordered mapping from every S2 stack index to its source scene/product ID.
2. **Dates and footprints:** NAIP acquisition timestamp and source footprint; each S2 acquisition timestamp and scene footprint; temporal-window definition used to include each frame.
3. **Per-chip georeferencing:** source and output CRS, affine transforms, exact AOI/crop bounds and pixel windows, tile x/y convention, pixel size, and any alignment/offset corrections for both products.
4. **Processing provenance:** source asset versions, TCI generation/normalization, reprojection software/version, resampling kernels, grid origin, NAIP downsampling kernel/phase, and cloud/NoData masks.

The paper suggests a metadata deliverable exists, but it is unknown whether the needed fields are in one compact downloadable table or embedded in a larger asset. The official Hugging Face repository is listed at 974 GB overall. Do not download that corpus or the broad archive until maintainers identify the precise metadata object, its URL/path, format, size, and small-val coverage. A compact CSV/JSON/Parquet table with the fields above would be appropriate; its availability and size remain unverified.

### Independent diagnostic after records are obtained

1. Freeze and checksum the original PNGs and source metadata; select the exact input S2 scene IDs used by inference.
2. Reconstruct each product’s map transform from authoritative CRS/affine/crop-window records and intersect their footprints. Project to one explicitly defined common CRS and grid using only documented transforms; retain resampling provenance and valid-data masks.
3. Quantify geometric residuals using independent control points or stable features in source imagery, not the ESRGAN output. Report per-tile residual vectors, robust summary/upper quantiles, overlap fraction, and invalid-pixel fraction. Define tolerances and pass/fail rules before inspecting final metric results; mark tiles failing them unscorable.
4. Join S2 frame dates to NAIP capture dates, apply a preregistered temporal bound, inspect cloud/NoData masks, and record exclusions and temporal gaps for each selected frame. Reconcile the README’s three-month and paper’s two-month descriptions with maintainer confirmation.
5. Only for tiles passing independent spatial and temporal screening, calculate pixel metrics on the verified common valid footprint. Preserve exploratory metrics separately. Do not optimize a flexible warp against the model prediction.

## 5. Recommendation and go/no-go

**GO** to request/locate the compact official provenance manifest and source-processing documentation. The paper’s metadata statement makes this a concrete, plausible recovery path.

**NO-GO** for claiming validated pixel-level performance, pixel registration, or building-level accuracy from the current `small_val_set` PNGs. Until per-image transforms, source crop windows, frame-to-scene/date mappings, and relevant cloud/temporal information are obtained and independently checked, retain existing PSNR/SSIM only as exploratory raw-array comparisons.

## Sources

- Local author documentation and implementation: `README.md`; `ssr/data/s2-naip_dataset.py`; `ssr/utils/infer_utils.py`; `ssr/infer.py`; `ssr/options/experiments/esrgan_baseline.yml`; `ssr/options/infer_satlas_full_validation.yml`.
- Official paper: [arXiv HTML, Sections 3 and 4.1](https://arxiv.org/html/2311.18082v1).
- Official code/data README and branch listing: [allenai/satlas-super-resolution, experiments branch](https://github.com/allenai/satlas-super-resolution/tree/experiments).
- Official dataset listing: [allenai/satlas-super-resolution on Hugging Face](https://huggingface.co/datasets/allenai/satlas-super-resolution), including its [validation tree](https://huggingface.co/datasets/allenai/satlas-super-resolution/tree/main/ssr-data/val_set).
- Separate SatlasPretrain documentation (used only to distinguish products): [SatlasPretrain.md](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md).
- Supplementary PDF linked by the official README: [direct PDF URL](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/Zooming_Out_On_Zooming_In_Supplementary.pdf); retrieval returned HTTP 401 during this investigation.

## Exact local files inspected and created

Inspected:

- `satlas-super-resolution/README.md`
- `satlas-super-resolution/ssr/data/s2-naip_dataset.py`
- `satlas-super-resolution/ssr/utils/infer_utils.py`
- `satlas-super-resolution/ssr/infer.py`
- `satlas-super-resolution/ssr/options/experiments/esrgan_baseline.yml`
- `satlas-super-resolution/ssr/options/infer_satlas_full_validation.yml`
- `satlas-super-resolution/reports/registration_provenance_investigation.md`
- `satlas-super-resolution/reports/satlas_sr_full_validation_evaluation.md`
- `satlas-super-resolution/results/satlas_sr_full_validation/summary.json`
- `data/small_val_set/` extracted file inventory and `data/super_resolution_small_val_set_2024-01-09.tar` (metadata/extension inventory and local size)

Created:

- `satlas-super-resolution/reports/satlas_sr_provenance_investigation.md`
