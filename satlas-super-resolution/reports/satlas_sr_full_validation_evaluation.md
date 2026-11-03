# Satlas-SR full-validation baseline evaluation and audit

## A. Verified

- Inference manifest contains 256 rows for 256 unique discovered Sentinel-2 tile directories. Statuses: {'OK': 256}. There are 0 duplicate manifest IDs, 0 duplicate/ambiguous NAIP reference IDs, and 0 missing/malformed-file or evaluation issues; missing TCI inputs, LR previews, predictions, and NAIP references were all 0.
- Exactly 256 tile IDs have one NAIP reference. The evaluation script checked each entry's input, LR preview, prediction, and reference for readability; all metrics rows that pass checks are RGB with matching 128x128 prediction/reference dimensions. The full output tree was also audited: every successful manifest entry points to a readable 128x128 RGB SR PNG and a readable 32x32 RGB LR-preview PNG.
- Full-run checkpoint: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\experiments\esrgan_8S2.pth`; SHA-256 `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`; weight key `params_ema`. Config: `ssr/options/infer_satlas_full_validation.yml`; seed `0`; eight TCI frames selected with the unchanged official `format_s2naip_data`; RGB values divided by 255; x4 output.
- Inference wall time `23.021` s; mean per-tile time `0.079` s (from the existing run summary). Observed uint8 ranges: LR preview [20, 255], SR [0, 255], NAIP [0, 255].
- The full-run predictions and LR previews for the five pilot IDs are byte-identical to pilot files. Selected frame indices match the saved reconstruction for each pilot tile: 20771_46525, 20890_46801, 20926_46045, 21068_45692, 21856_49804.
- Local inference uses the repository's released 8-image configuration (`SSR_RRDBNet`, 24 input channels, 3 output channels, scale 4, 64 features, 23 RRDB blocks, growth 32). The instantiated model/checkpoint has 16,710,083 parameters, consistent with the paper's approximately 17M small ESRGAN. The official config uses `params_ema`.
- Procedure difference: this custom runner sorts tile directories before consuming one seeded Python RNG; stock `ssr.infer.py` traverses recursive `glob.glob()` results without an explicit sort. Because frame choice uses that RNG, enumeration order can change per-tile frames. The five pilot IDs occur first in this sorted full run and match their recorded indices and output hashes; all full-run indices are in the manifest.

## B. Exploratory

- Raw PSNR/SSIM were computed for 256 tiles after checking RGB mode, 128x128 dimensions, readability, and uint8 range. These numbers are marked `METRICS_COMPUTED_EXPLORATORY` in `tile_metrics.csv`; excluded rows remain in the CSV with reasons. PSNR mean/sample SD: 18.8122 / 3.8138 dB; SSIM mean/sample SD: 0.3950 / 0.1495. data_range=255; SSIM settings: `channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True`.
- The contact sheet shows 12 deterministic representative tiles (the five pilot tiles plus seven spread across sorted remaining IDs). Its LR panel shows the eight selected S2 frames, in manifest order, as a 4x2 montage; thumbnails use nearest-neighbor display enlargement only. SR and NAIP panels remain at 128x128.
- Interpret these as array comparisons only. The repository says the images use a shared Web-Mercator zoom-17 tile system and are intended to overlap spatially; the loader pairs by tile ID. The archive contains PNGs without CRS/affine/pixel-origin/crop-window metadata. The README's generic S2 reproject example is not per-image proof of an identical grid. Therefore matching tile IDs and equal dimensions do not verify pixel registration.
- Temporal documentation differs: the checked-out README says temporal overlap within three months, while paper Section 4.1 says overlapping S2 imagery within two months. This archive has NAIP date tokens in its folder UUIDs but no dates/source scene IDs per S2 stack frame. The paper says clouds are not removed, so cloud and scene changes can affect comparisons. The discrepancy and missing per-frame metadata prevent validating exact temporal offsets for these image pairs.
- The paper states that geospatial and temporal metadata are provided for all imagery (Section 4.1), but those records are not present in the extracted small_val_set archive and were not located in the checked-out project. They may be separately distributed; they should be recovered before registration validation.

## C. Not yet established

- Scientifically valid pixel-level PSNR/SSIM or the paper's cPSNR/LPIPS/CLIPScore results: registration and radiometric/temporal comparability are not established here. This run computes standard PSNR/SSIM only; it does not reproduce the paper's full metric suite or human study.
- Building reconstruction accuracy, missed-building rates, hallucination rates, or building-level generalization. No labels or spatially verified annotations were created.
- Exact paper reproduction. This work ran the released pretrained 8S2 checkpoint on the 256-image urban small validation subset, without training. It does not reproduce training, the paper's full data split/experiment, the reported medium 87M ESRGAN in its method comparison, or the paper's CLIPScore/cPSNR/LPIPS and expert building-count protocols. The instantiated released checkpoint is the 17M small model.

## Files and rerun

- Per-tile results and exclusions: `results/satlas_sr_full_validation/tile_metrics.csv`.
- Machine-readable audit and aggregate metrics: `results/satlas_sr_full_validation/summary.json`.
- Representative 3-panel contact sheet: `results/satlas_sr_full_validation/representative_contact_sheet.png`.
- Re-run from repository root with the existing environment: `python evaluate_satlas_sr_validation.py`.
- Inference was not rerun for this evaluation.

## References

- Repository documentation/configuration: `README.md`, `ssr/options/experiments/esrgan_baseline.yml`, `ssr/data/s2-naip_dataset.py`, `ssr/utils/infer_utils.py`.
- Wolters, Bastani, Kembhavi, *Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing*, arXiv:2311.18082, Sections 3, 4.1, 5 and building-count study: https://arxiv.org/html/2311.18082v1.
