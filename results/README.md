# Benchmark Results & Experiment Artifacts

This directory stores frozen benchmark manifests, per-tile metric CSV files, prediction images, and audit logs across all research iterations.

## Directory Structure & Highlights

| Subdirectory / File | Description |
| :--- | :--- |
| `satlas_sr_baseline_freeze_v1/` | Frozen baseline artifact directory containing `baseline_manifest.json`, environment logs, and baseline reproduction metrics. |
| `satlas_sr_revalidation/` | Per-tile evaluation CSV (`per_tile_metrics.csv`) containing cPSNR, PSNR, SSIM, LPIPS, and CLIPScore across all 256 validation tiles. |
| `satlas_sr_pilot_five_tiles/` | Super-resolved PNG predictions and inference log for the 5 pilot smoke-test tiles. |
| `satlas_sr_full_validation/` | Output directory containing inference results and metrics across the entire 256-tile urban validation set. |
| `satlas_sr_paper_metrics/` | Official paper reproduction benchmark results formatted in JSON and CSV. |
| `satlas_sr_experiment_001/` ... `024/` | Individual experiment run folders capturing the research progression from baseline audits (EXP-001) to uncertainty calibration (EXP-015) and soft-blending reconstruction (EXP-023). |
| `MASTER_PROJECT_COMPREHENSIVE_REPORT.md` | Local copy of the comprehensive 24-experiment master report. |

