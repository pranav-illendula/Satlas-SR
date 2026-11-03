# Satlas-SR validation provenance and registration investigation

**Scope.** Read-only investigation of the existing repository, extracted small validation archive, five-tile audit artifacts, and dataset/inference implementation. No inference was run; no image or other existing project file was changed. The report file is the only intended addition.

## Verified findings

### Provenance and pairing in the repository

- The repository README (`README.md:93-118`) describes Sentinel-2/NAIP data as image pairs; says both products use the SatlasPretrain Web-Mercator tile system; defines NAIP paths as `naip/{image_uuid}/{tile}/rgb.png`, where `image_uuid` carries the NAIP capture timestamp and `{tile}` identifies a cell in a `2^17 x 2^17` grid; and defines S2 paths as `sentinel2/{tile}/{band}.png`. The README says each NAIP image has a corresponding S2 time series, and explicitly says S2 stack order need not be chronological.
- `ssr/data/s2-naip_dataset.py:114-126` extracts the parent tile directory from each NAIP path and constructs S2 band paths using that exact tile key. This establishes the dataset's intended *tile-level association*. It does not compare coordinates, pixel origins, transforms, or image content to establish pixel registration.
- The same loader reads the NAIP chip and reshapes stacked S2 frames at `ssr/data/s2-naip_dataset.py:176-240`; it can select random valid frames. Random crop/resizing is conditional on `rand_crop` (`:56`, `:232-240`) and is commented as training augmentation. The official baseline validation/test config does not set `rand_crop` (`ssr/options/experiments/esrgan_baseline.yml:40-70`). The separate inference path reads PNGs and passes them to `format_s2naip_data` (`ssr/infer.py:42-54`); that helper reshapes the stack, chooses valid frames, and divides RGB values by 255 (`ssr/utils/infer_utils.py:6-38`). Neither inference code nor this validation loader georeferences or warps the S2/NAIP pair.
- The README includes an example for preparing *raw* Sentinel-2 TCI data: reproject to EPSG:3857 at 9.555 m resolution with bilinear resampling (`README.md:125-132`). This is general preprocessing guidance, not per-chip metadata or evidence that the archived NAIP chips and S2-derived PNGs have matching pixel origins/resampling histories. The repo file search found no dataset-generation script or per-tile transform manifest in the checked-out project.

### Upstream documentation checked

- The current upstream [Satlas-SR README](https://github.com/allenai/satlas-super-resolution/blob/main/README.md#s2-naip-dataset-structure) adds a useful dataset-level statement: each S2 time series is described as overlapping its NAIP image spatially and temporally within three months. It also describes NAIP chips as 128x128 pixels at 25% of original NAIP resolution and the S2 inputs as 32x32 frames stacked vertically. This supports intended dataset pairing and a coarse temporal bound, but does not provide per-observation dates, transforms, crop windows, or registration residuals for this extracted validation subset. The documentation also warns that stacked S2 frames need not be chronological.
- The upstream README's raw-TCI reprojection example remains illustrative processing guidance; it does not say that those exact parameters, source scenes, or pixel-grid origins apply to each archived validation chip. The checked-out repository has no linkable per-tile manifest or source-scene mapping that would close this gap.
- The similarly named, newer [AI2-S2-NAIP Hugging Face dataset card](https://huggingface.co/datasets/allenai/s2-naip) describes a distinct dataset format with UTM GeoTIFFs and scene-ID tables. Its metadata must not be transferred to this Satlas-SR `small_val_set`; I found no evidence that this archive was generated from that dataset.

### Archive, metadata, and dates

- The extracted dataset inventory is 2,816 PNG files and no other file extensions. TAR inventory is 2,816 `.png` members plus directory entries; no JSON/CSV/XML/world files or raster source files were present. Existing audit inspection of the five TCI/output/reference triples found only PNG `IHDR`, `IDAT`, `IEND` chunks, no Pillow metadata or EXIF, and no geospatial tags. Thus the delivered files provide no CRS, affine transform, coordinates, pixel size, or image registration parameters.
- NAIP folder UUIDs do expose date-like tokens, and the README specifically says `image_uuid` has the NAIP capture timestamp. For the five audited tiles the folder tokens recorded in `reports/five_tile_evaluation_audit.md` are `20191010`, `20190805`, `20190826`, `20191011`, and `20190812` respectively. These identify NAIP-side dates as encoded in the folder names; no source record here confirms timestamp precision beyond that convention.
- No timestamp, source scene ID, or chronological mapping is present for any stacked S2 frame. The reconstructed indices in `reports/reconstructed_tci_frame_indices.csv` identify positions in the PNG stack selected by the inference helper, not dates. The README's warning that frame order need not be chronological prevents treating stack index as time.
- Tile IDs (`20771_46525`, etc.) are shared grid-cell identifiers, not latitude/longitude or pixel transform metadata. Matching tile IDs and dimensions do not verify that the NAIP crop and any particular S2 observation cover precisely the same pixel footprint or are subpixel aligned.

### Registration conclusion and current evaluation suitability

The current PNG package does **not** support a defensible pixel-level registration verification: coordinates, CRS/affines, pixel origin, pixel size, resampling/cropping provenance, and S2 frame dates are absent. No independent registration record was found in the checked-out source. Existing equal dimensions and tile names alone are insufficient. Therefore the existing raw PSNR/SSIM can be retained as exploratory same-tile comparisons, but should not be described as validated pixel-level quality scores.

The package also does **not yet support defensible building-level error analysis**. Building matching requires an established spatial transform and temporally appropriate reference evidence; neither is available in these PNGs. Visual inspection can raise candidates for human investigation, but cannot establish missed, shifted, or hallucinated buildings. No labels were created. The dataset is currently suitable for exploratory visual comparison and provisional raw image-pair metrics only; for inferential pixel metrics and building-level claims, it is presently **neither**.

## Registration test options

No geospatial sidecar is available for a minimal transform-based test on the existing five tiles, so none was attempted and no coordinates were inferred. If upstream source assets or metadata can be recovered, preserve the original PNGs and create a separate diagnostic: obtain source CRS/affines and exact chip windows for both products; project them into a common CRS/grid using the documented source transforms; calculate corner/landmark residuals and valid-data overlap; inspect independent stable features; record transform/resampling and residuals per tile; and predefine an acceptance threshold before metrics. Do not estimate a flexible warp by maximizing similarity to the ESRGAN output.

The upstream documentation narrows the provenance request but does not resolve it. The next useful action is to request from the Satlas-SR maintainers (the README lists an issue tracker and contact email) the small validation archive's per-NAIP identifier to NAIP source scene/date mapping, per-tile crop bounds/transforms, per-S2-frame scene IDs and acquisition times, and the actual reprojection/resampling/cropping steps used for this archive. If those records cannot be obtained, select a smaller evaluation subset directly from source imagery that retains authoritative georeferencing and sufficiently close acquisition dates, then verify registration independently before scaling. The published dataset-level overlap statement is not a substitute for per-observation records.

## Proposed building annotation protocol (not ground truth)

Annotate only after registration is accepted and each image is judged temporally/visually comparable. Two annotators independently draw roof-footprint polygons in both output and reference, blinded to PSNR/SSIM. Record tile ID, polygon, source panel, visibility, confidence, and reason codes. Pre-register a one-to-one match rule, for example polygon IoU >= 0.5, and calibrate it on a separate adjudicated pilot before production.

- **Correct:** a visible reference roof has a one-to-one output roof match under the preregistered rule and location tolerance.
- **Missed:** a clearly visible reference roof has no matching output roof.
- **Shifted:** the same recognizable roof is present in both, but the output footprint exceeds the accepted location tolerance; annotate displacement separately. If the whole tile fails registration, mark the tile unscorable rather than counting every roof as shifted.
- **Potentially hallucinated:** a building-like output footprint has no reference match after tile registration and plausible temporal change have been checked against independent, date-appropriate evidence. Without that evidence, use ambiguous/output-only candidate, not hallucinated.
- **Ambiguous:** partial/occluded roofs, cloud or shadow, low resolution, uncertain boundary, temporal construction/demolition, poor registration, or annotator disagreement. Keep a reason code; adjudicate disagreements and retain unresolved items as ambiguous rather than forcing a class.

These are proposed operational definitions only. They are not labels observed or produced in this investigation.

## Unresolved questions and recommended next actions

1. Obtain the upstream per-image geospatial provenance: NAIP acquisition date/time and footprint; Sentinel-2 scene IDs and acquisition times for each stack frame; CRS, affine transforms, tile bounds/windows, pixel size, cropping, resampling, and any alignment corrections.
2. Confirm whether the README's 9.555 m EPSG:3857 resampling example was applied to these exact S2 sources, and how the 128x128 NAIP chips were cropped/resampled to corresponding tile bounds.
3. With provenance in hand, run a separate five-tile registration diagnostic and document per-tile residuals, valid overlap, and pass/fail; do not alter the originals.
4. Only after registration and temporal screening, freeze image-processing/metric settings, retain all 256 rows with rejection reasons, and pilot dual-annotator building labels with adjudication and agreement reporting.

The upstream documentation search was performed on 2026-10-19. It did not involve contacting maintainers or downloading a different dataset.

## Reproduction / evidence commands

Run from PowerShell at the repository root (paths below use the user's existing layout). These are inspection commands; they do not rerun inference or change source images.

```powershell
# Repository and available source files
git status --short --branch
rg --files | Select-String -Pattern 'generate|process|download|metadata|georef|raster|dataset|infer'

# Locate the source statements cited above
Select-String -Path README.md -Pattern 'Web-Mercator|image_uuid|time series|reproject_image|resolution='
Select-String -Path ssr\data\s2-naip_dataset.py -Pattern 'rand_crop|split_path|s2_paths|random.sample|F.interpolate'
Select-String -Path ssr\utils\infer_utils.py -Pattern 'def format|np.reshape|random.sample|/255'
Select-String -Path ssr\infer.py -Pattern 'glob.glob|skimage.io.imread|format_s2naip_data'

# Inventory extracted files and archive members
Get-ChildItem 'C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set' -Recurse -File |
  Group-Object Extension | Select-Object Name,Count
tar.exe -tf 'C:\Users\CSE-SDPL\Desktop\DL_image\data\super_resolution_small_val_set_2024-01-09.tar' |
  ForEach-Object { if ($_ -match '\.') { [IO.Path]::GetExtension($_).ToLowerInvariant() } else { '[directory/no extension]' } } |
  Group-Object | Sort-Object Name | Select-Object Name,Count

# Review prior visual/provenance evidence
Get-Content reports\five_tile_evaluation_audit.md
Get-Content reports\reconstructed_tci_frame_indices.csv
```

`git status` at investigation time showed pre-existing modifications/untracked work including `ssr/infer.py`, `experiments/`, `reports/`, `results/`, and the pilot scripts/configs. These were not edited by this investigation.
