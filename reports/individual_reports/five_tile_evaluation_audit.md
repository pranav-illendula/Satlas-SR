# Five-tile Satlas-SR evaluation audit

**Scope:** reviewed the existing five-tile inference/evaluation scripts, configuration, logs, metrics, inputs, outputs, and NAIP references. No model inference was rerun and no source images were modified.

## What the files establish

- The five tile IDs are `20771_46525`, `20890_46801`, `20926_46045`, `21068_45692`, and `21856_49804`. The official helper received one TCI stack per ID and selected eight of its 32 frames. All 32 frames in each audited stack passed its non-black check.
- Each TCI PNG decodes to 32 x 1024 pixels (array shape 1024 x 32 x 3), RGB. Each ESRGAN output and matched NAIP reference is 128 x 128 RGB. All inspected files are valid PNGs.
- For all 15 TCI/output/reference PNGs, Pillow reports no metadata or EXIF tags; raw PNG chunks are only `IHDR`, `IDAT`, and `IEND`. The extracted validation dataset contains PNGs only, with no georeferencing sidecars. Thus these files provide no CRS, affine transform, coordinate origin, pixel size, or acquisition tags.
- The repository README says both products use the same Web-Mercator tile system and documents the NAIP folder as `naip/{image_uuid}/{tile}/rgb.png`; the dataset loader finds the Sentinel-2 tile by that same tile ID. This is tile-level association, not evidence of subpixel or pixel-level registration. Matching names and 128 x 128 dimensions do **not** verify alignment.
- The existing metric CSV is therefore a raw, alignment-unverified comparison. Treat its PSNR/SSIM values as provisional image-pair scores, not validated pixel-quality results.

The visual audit shows the eight selected TCI frames, ESRGAN output, and tile-matched NAIP side by side. Frame indices were reconstructed after the run by replaying seed 0, sorted tile order, and the unmodified official `format_s2naip_data` helper. The inference log did not persist frame indices; the reconstruction is documented in [reconstructed_tci_frame_indices.csv](reconstructed_tci_frame_indices.csv).

![Five-tile visual audit](five_tile_visual_audit.png)

## Tile-by-tile visual notes

These are visual observations from the audit sheet, not object labels. Cloud/building interpretations are uncertain at this display scale.

| Tile | NAIP folder date token | Visible observation and metric risk |
|---|---|---|
| `20771_46525` | `20191010` | Both show a green, textured landscape with lighter developed/open patches. Fine texture and local contrast differ. The selected TCI frames look dark and low-detail; no building-level conclusion is supported. |
| `20890_46801` | `20190805` | Both show a near-vertical corridor and a broad open field. ESRGAN appears smoother and lighter in parts. Several selected TCI frames contain bright, cloud-like areas; cloud identity is not confirmed. |
| `20926_46045` | `20190826` | The panels show broad diagonal/curved transport features, but local road and vegetation patterns are not an obvious pixel-for-pixel match at sheet scale. This warrants registration review; it does not by itself prove a shift. Selected TCI frames include bright/cloud-like content. |
| `21068_45692` | `20191011` | Both are dominated by dense vegetation. Canopy texture and the locations/contrast of darker gaps differ. The selected TCI frames are mostly dark/low-detail. |
| `21856_49804` | `20190812` | Both show open land with scattered dark features. ESRGAN looks smoother and more uniform; NAIP has more high-contrast small patches. Selected TCI frames include pale/bright, cloud-like content. |

The NAIP folder IDs contain these date tokens; the README describes `image_uuid` as carrying the capture timestamp. No date for any individual Sentinel-2 frame can be established from the stacked PNGs. The README identifies Sentinel-2 L1C time series and says frame order need not be chronological. The S2-to-NAIP time gap, season, cloud conditions, and intervening construction or land-cover changes are therefore unknown. Such differences, along with registration, can affect PSNR/SSIM. The pale or bright TCI areas above are confirmed visual properties; calling them clouds is an interpretation.

## Metric record

The existing `five_tile_metrics.csv` reports all five tiles as successful:

| Tile | PSNR (dB) | SSIM |
|---|---:|---:|
| `20771_46525` | 19.5990 | 0.3072 |
| `20890_46801` | 19.7629 | 0.4119 |
| `20926_46045` | 19.2156 | 0.3248 |
| `21068_45692` | 20.8481 | 0.3021 |
| `21856_49804` | 17.5323 | 0.4079 |
| **Mean** | **19.3916** | **0.3508** |
| **Sample standard deviation** | **1.2031** | **0.0546** |

These are uint8 comparisons (`data_range=255`). SSIM used `channel_axis=-1`, `win_size=7`, `gaussian_weights=False`, and `use_sample_covariance=True`. Their interpretation remains conditional on alignment and temporal comparability. This five-tile selection is a small convenience pilot, not a random or stratified sample of the 256 urban pairs.

## Recommended protocol before evaluating all 256

1. Recover georeferenced source rasters or authoritative sidecars for both products, including CRS, transforms, pixel origin/resolution, and per-frame Sentinel-2 acquisition times. Preserve the original PNGs.
2. Derive a documented transform from source geospatial metadata, then verify residual alignment with independent landmarks/edges and manual review. Set an acceptance tolerance before scoring. Do not align by optimizing against the ESRGAN output; do not use flexible local warps that can hide model errors. Record transform, residual, valid masks, and a pass/fail reason for every tile. Keep rejected rows in outputs.
3. Separate pixel-comparable pairs from temporally/observationally ambiguous pairs (cloud, shadow, seasonal change, construction, or unmatched dates). Report counts and reasons. For eligible pairs, fix the registration/resampling and color/range protocol in advance, then report per-tile PSNR/SSIM plus aggregate distributions and a defined baseline. Keep every tile in the manifest; never silently drop failures.
4. Run preflight checks across all 256 pairs: unique tile mapping, file readability/mode/shape, black/no-data content, valid S2-frame counts, and date/provenance availability. Persist the selected eight frame indices per tile alongside seed and tile order.
5. Treat building-level review as a separate, human-annotated outcome. After registration is approved, two independent annotators should polygon-label visible roof footprints in reference and output, blinded to metric scores, with confidence and reason codes. A proposed match rule is footprint IoU >= 0.5, calibrated on a pilot annotation set before production. A **missed-building candidate** is a reference-confirmed visible roof with no matched output roof. An **output-only building candidate** is a building-like output footprint without a reference match; call it a *potential hallucination* only after temporal change and registration are ruled out using time-appropriate independent evidence. Mark partial roofs, canopy/shadow occlusion, low-resolution objects, temporal change, and annotator disagreement as ambiguous/unscorable; adjudicate disagreements and do not force binary labels.

No building labels were created here. This pilot does not establish hallucination detection, uncertainty calibration, generalization, or research novelty.

## Verified and unresolved

**Verified:** tile IDs map to unique NAIP references; dimensions/modes and PNG readability; no geospatial metadata in inspected PNGs; official helper selection behavior and a deterministic reconstruction of selected frames; existing per-tile metrics and visual differences. The inference log records five successes, no failed tiles, per-tile times of 0.500, 0.085, 0.090, 0.078, and 0.107 seconds, and 73.60 MiB peak PyTorch allocation per tile. Recorded non-fatal warnings: torchvision transforms deprecation, SciPy `ndimage.filters` deprecation, and a low-contrast warning for the `20890_46801` LR preview.

**Unknown / resolve before full evaluation:** pixel registration and geospatial transforms; Sentinel-2 per-frame dates and S2/NAIP temporal offsets; acquisition-condition comparability; whether visible local differences are model error, registration, or scene change; building-level truth and annotation agreement. These are blockers to treating raw PSNR/SSIM or output-only structures as validated quality/hallucination findings.