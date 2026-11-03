# Satlas-SR full-validation inference run

- Run completed (UTC): 2026-10-15T11:50:24.536707+00:00
- Config: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\ssr\options\infer_satlas_full_validation.yml`
- Dataset: `C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set\sentinel2`; NAIP refs: `C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set\naip`
- Checkpoint: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\experiments\esrgan_8S2.pth` (SHA-256 `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`; key `params_ema`)
- Device: NVIDIA RTX 2000 Ada Generation; PyTorch 2.1.0+cu121; CUDA available: True
- Discovered in this run: 256; matched to exactly one NAIP reference: 256
- Status counts: OK 256, skipped valid 0, failed 0, pending 0
- Total elapsed time: 23.021 seconds
- Mean per-tile inference time (OK rows only): 0.079 seconds
- Output directory: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\results\satlas_sr_full_validation`
- Incremental manifest: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\results\satlas_sr_full_validation\inference_manifest.csv`

## Reproducibility

The official `format_s2naip_data` helper is used unchanged. Python random seed is set once to `0`; tile directories are traversed in sorted ID order, and frame indices are captured by observing the helper's existing `random.sample` calls. Inputs are RGB TCI stacks divided by 255 by the official helper; the configured generator uses 8 S2 frames, 24 input channels, scale 4, and checkpoint key `params_ema`. Outputs are clamped to [0,1], converted to uint8 RGB PNGs, and expected to be 128x128. This reproduces the released inference pipeline; it does not verify geospatial registration or temporal alignment and makes no accuracy claim.

Failures remain in `inference_manifest.csv`; no NAIP-based PSNR/SSIM is calculated by this run.
