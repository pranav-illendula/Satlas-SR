# Satlas-SR base-paper reproduction feasibility audit

**Audit date:** 2026-10-19  
**Scope:** Read-only feasibility assessment of the current Satlas-SR checkout, available local data/checkpoint/results, environment, and the official paper. This report is the only new file. No inference, evaluation, or training was run for this audit; no existing project file, result, or environment was changed.

## Executive summary

The available work **reproduces inference from the released small (about 17M-parameter), eight-Sentinel-2-frame ESRGAN checkpoint on all 256 tiles in the small validation subset**. The run is recorded with checkpoint/config hashes, selected frame indices, seed, and output paths. This is a useful checkpoint-inference reproduction and a qualitative/exploratory evaluation. It is **not a reproduction of the paper’s training or complete experimental results**.

Three independent gaps prevent stronger claims:

1. The paper’s principal four-method comparison uses a **medium, 87M-parameter ESRGAN**, while the local released checkpoint/configuration is the **small, 17M-parameter model**. The paper’s primary S2-NAIP metric is CLIPScore, alongside cPSNR and LPIPS; the local 256-tile evaluation reports standard PSNR and SSIM only.
2. The local PNG subset does not establish per-frame Sentinel-2 source IDs/dates, exact source/crop transforms, or pixel registration. The PSNR/SSIM values therefore remain exploratory array comparisons, not validated pixel-level results.
3. The local checkout has no training corpus, no SRCNN/HighResNet/SR3 comparison checkpoints, no SR3 implementation, and no downstream task datasets/annotation outputs. The one-GPU environment has 16 GiB VRAM, but no training throughput or memory benchmark was run. The baseline YAML specifies batch size 256 per GPU and 100,000,000 iterations, so full training feasibility cannot be claimed from the successful short inference run.

**Recommendation:** GO for preserving and describing the existing checkpoint inference and visual audit as a limited reproduction. NO-GO for claiming faithful reproduction of the paper’s numerical findings, validated PSNR/SSIM, or the full paper. First obtain the official supplement’s exact experimental recipes and the compact provenance records covering the local validation subset; then select one paper experiment and freeze its data split, model, metric, and acceptance criteria before any new compute.

## Evidence basis and labels

- **Paper-documented:** explicitly stated in the official paper’s main text. Detailed training settings are referred to Supplement A.3, and other experiment details to A.1–A.6; those cannot be treated as absent merely because the linked PDF was not available through the read-only browser path used for the prior provenance investigation.
- **Locally verified:** recorded in a local source/config/result/report or established by read-only file/environment inspection.
- **Inference:** a conclusion from comparing those sources; it is labeled as such and not represented as an author statement.

The paper itself says geographic and temporal information is provided for the S2-NAIP imagery (Section 4.1). The evidence gap is that this information was not found in the extracted small validation archive or the checked-out repository. It may be distributed separately. The repository README also says the S2 series overlaps NAIP within three months, whereas paper Section 4.1 says two months; resolve this difference with the authors or authoritative metadata rather than choosing one silently.

## Experiment feasibility matrix

| # | Experiment and what the paper did | Local evidence and prerequisites | Status and smallest next action |
|---|---|---|---|
| 1 | **Released small ESRGAN inference.** Apply a trained 17M small ESRGAN to eight S2 TCI frames and produce 4× RGB outputs. | `results/satlas_sr_full_validation/run_summary.md`, `inference_manifest.csv`, and `summary.json` record 256/256 successful tiles, `params_ema`, seed 0, chosen frame indices, hashes, and 128×128 outputs. Config is `ssr/options/infer_satlas_full_validation.yml`; checkpoint is `experiments/esrgan_8S2.pth`. The model family/settings match the small 8-frame configuration in `ssr/options/experiments/esrgan_baseline.yml`; instantiated parameter count recorded as 16,710,083. | **READY WITH LIMITATIONS.** Preserve this as released-checkpoint inference on the available small subset. It does not establish how the checkpoint was trained or reproduce the paper’s main medium-model experiment. No rerun needed. |
| 2 | **Paper-intended validation/test evaluation.** The paper compares held-out data and reports across four datasets; S2-NAIP has 1.2M pairs. | Only `data/small_val_set` and derived pilot subsets are present locally; no full train, 8,192-pair `val_set`, or paper test set was found. The small set has 256 pairs and PNGs only. Provenance reports record no S2 frame dates/scene IDs or exact transforms/crop windows. | **BLOCKED BY DATA/PROVENANCE.** Obtain the split identity and compact authoritative provenance manifest; then identify/download only the named evaluation split after its source, scope, and size are confirmed. |
| 3 | **PSNR and SSIM.** These are reported in the repository test config for compatibility, but the paper emphasizes their limitations and prioritizes CLIPScore. | `evaluate_satlas_sr_validation.py` computes RGB raw-array PSNR (`data_range=255`) and SSIM (`channel_axis=-1`, window 7, Gaussian weights false, sample covariance true) for all 256. It uses no crop border, whereas `ssr/options/experiments/esrgan_baseline.yml` sets crop border 4. `results/.../summary.json` explicitly marks registration unverified. | **READY WITH LIMITATIONS.** Computation has been done, but scores are exploratory only and not the paper’s primary metric. For validated pixel-level metrics, first verify transforms, temporal suitability, valid masks, and paper’s exact protocol; do not compare current values as validated performance. |
| 4 | **LPIPS/perceptual metrics.** Paper Table 2 reports LPIPS-derived preference agreement (higher is better after conversion), as well as CLIPScore and cPSNR. | `ssr/metrics/lpips.py` and test configuration exist; `lpips` is importable in the current venv. But no LPIPS result was produced for this run, exact paper conversion/evaluation details are delegated to the supplement, and pairwise image comparability is still unverified. | **BLOCKED BY DATA/PROVENANCE.** First obtain authoritative spatial/temporal provenance and supplement’s LPIPS protocol; then calculate only on an approved evaluation split. |
| 5 | **CLIPScore and human-metric correspondence.** Paper defines image-embedding cosine similarity; CLIPA-v2 ViT-bigG-14/DataComp1b is the best metric in its human study (84.6% agreement). | `ssr/metrics/clipscore.py` supports CLIPA via `open_clip`, but the module imports `clip` unconditionally; `clip` is absent in the venv (`find_spec` false), though `open_clip` is present. Pretrained weight availability/cache was not checked. No local human preference annotations exist. | **BLOCKED BY IMPLEMENTATION.** Reconcile the unconditional dependency in the metric path and acquire the exact permitted model weights/protocol before a small metric smoke check. Human-agreement replication separately requires the preference dataset/study records. |
| 6 | **SRCNN, HighResNet, ESRGAN, SR3 comparison.** Paper Section 5 compares four methods on S2-NAIP, PROBA-V, WorldStrat, and OLI2MSI; it uses 87M medium ESRGAN to compare with 97M SR3. | Local architecture and dataset code/configs exist for SRCNN and HighResNet; ESRGAN exists. The README says SR3 code lived in a separate repository and was being prepared for release. Only local checkpoint is `esrgan_8S2.pth`; no other method weights or comparison outputs were found. Official test config includes cPSNR/LPIPS/CLIPScore, but current results are only PSNR/SSIM. | **BLOCKED BY IMPLEMENTATION.** Obtain the exact supplement recipe plus accessible SR3 implementation and the comparison checkpoints or train recipes. For the smallest start, select one dataset/method comparison and verify each method can load before any evaluation. |
| 7 | **Dataset-size scaling.** Paper trains on 1%, 3%, 10%, 30%, and 100% of S2-NAIP and reports performance scaling. | YAMLs exist for 1%, 3%, 10%, and 30% variants (for example `ssr/options/experiments/esrgan_baseline_1perc.yml` through `_30perc.yml`); the base config points at an urban training set and weights JSON not present locally. No training logs/checkpoints for these runs were found. | **BLOCKED BY DATA/PROVENANCE.** Obtain the exact paper split/sample manifests and local access to the corresponding training subset; verify no leakage and preserve split IDs before training. |
| 8 | **Small/medium/large ESRGAN comparison.** Paper defines 17M, 87M, and 347M model sizes and crosses them with the five data fractions. | Local small config/checkpoint is present; `esrgan_larger1*.yml` and `esrgan_larger2*.yml` are present, but no medium/large checkpoint or training data is local. Available GPU is one NVIDIA RTX 2000 Ada Generation with 16,380 MiB reported VRAM. There is no batch/memory/throughput benchmark. | **BLOCKED BY DATA/PROVENANCE.** Get the exact model definitions and training data/splits first. Then benchmark memory/throughput on one documented batch size; do not infer full-scale training feasibility from 256-tile inference. |
| 9 | **CLIP-based auxiliary-loss ESRGAN.** Paper Section 6 trains the small ESRGAN from scratch with an L1 distance on CLIP features and reports a 9-point CLIPScore gain in 18× less training time; the best model adds further domain knowledge. | CLIP-loss YAMLs exist (`ssr/options/experiments/esrgan_cliploss.yml`, `esrgan_baseline_cliploss.yml`), but `clip` dependency is absent. Paper says this training used ResNet50 CLIP, while CLIPScore evaluation uses CLIPA; exact optimization/details are in supplement. No trained CLIP-loss checkpoint/logs found. | **BLOCKED BY IMPLEMENTATION.** Retrieve Supplement A.5 and exact CLIP loss implementation/dependency requirements; confirm a reproducible training recipe and available weights before training. |
| 10 | **Downstream machine-task evaluation.** Paper evaluates SR imagery and transferred SR features on six tasks (three Satlas tasks, PASTIS, BigEarthNet, and an OSM urban dataset). | The repository’s SR code is present, but no task-specific evaluation pipeline, task data/labels, or paper run artifacts were found locally. The paper’s task models and splits are separate prerequisites. | **BLOCKED BY IMPLEMENTATION.** Select one task and retrieve its exact paper data/split, model recipe, and scoring script; reproduce a low-resolution baseline before adding SR inputs. |
| 11 | **Building-count and hallucination analysis.** Paper uses expert annotations on a held-out 256-example set to compare ESRGAN and SR3 building correctness/hallucinations. | Existing 256 small-val results are not documented as the paper’s held-out building subset; no labels, annotation guide, SR3 predictions, or adjudication records were found. Local provenance also cannot yet establish precise temporal/spatial comparability. | **BLOCKED BY DATA/PROVENANCE.** Obtain the held-out tile list, expert annotations/definitions, and source provenance; otherwise this is a new human annotation study, not reproduction. |
| 12 | **Human preference evaluation.** Paper collects 11,524 pairwise judgments over generated outputs on 20,000 S2-NAIP and 381 WorldStrat datapoints; supplementary A.1 describes details. | No preference labels or participant protocol/results are local. Full recreation needs comparable multi-method outputs, target data, a study design, and human judgments. | **NOT PRACTICAL WITH CURRENT RESOURCES.** Prefer locating the original released labels/protocol; do not launch a new human study as a prerequisite for the limited baseline inference reproduction. |
| 13 | **Other major results: DDIM/CFG, domain-knowledge “best” model, global deployment.** Paper reports DDIM and classifier-free-guidance variants, combines CLIP loss with domain knowledge for a 0.959 CLIPScore best model, and describes global deployment. | The local ESRGAN code/configs do not reproduce these experiments as a set. Paper directs DDIM/CFG and domain-knowledge specifics to Supplement A.4/A.5. No global model/checkpoint or deployment pipeline is identified in the local evidence. | **BLOCKED BY IMPLEMENTATION.** Obtain supplement and the exact artifacts/pipeline for a selected claim. Treat global deployment as out of scope for a local base-paper reproduction unless a deployment reproduction is explicitly required. |

### What is matched, and what is not

- **Matched at inference level:** the released small 8-S2 ESRGAN family, expected 24-channel RGB input, 4× scale, EMA weights, seed/frame-selection record, and 128×128 output size. The run processed the full available 256-tile subset.
- **Not matched to the paper’s principal method-study ESRGAN:** paper Section 5 specifies the 87M medium model for comparison with SR3; this run uses the 17M small checkpoint.
- **Not a training reproduction:** a config is not a run record. The local YAML gives a candidate training recipe (Adam G/D, both lr 1e-4, betas 0.9/0.99, EMA 0.999, 100M iterations, L1 + VGG19 perceptual + vanilla GAN losses, one D iteration), but there is no local training log/history that proves those settings produced this checkpoint. Main-paper specifics refer to Supplement A.3.
- **Not a full numerical-result reproduction:** current PSNR/SSIM are exploratory and use a local protocol that differs from config crop-border settings. The paper’s Table 2 reports cPSNR, LPIPS-derived agreement, and CLIPScore, not this current pair of mean statistics. Do not compare the values directly.

## Data and provenance constraints

### Locally established

- `data/small_val_set` has `naip/` and `sentinel2/` trees and a 2,816-file extracted inventory containing PNGs only. The inference/evaluation manifests document 256 matched TCI/NAIP pairs. The three dataset directories present under `data/` are `small_val_set`, `small_val_set_tci_five_tiles`, and `small_val_set_tci_pilot`; no training, full validation, or paper test data directory was found.
- The README documents intended shared Web-Mercator zoom-17 tile keys, NAIP chips at 128×128 (one quarter original resolution), S2 RGB TCI stacks stored as 32×32 frames, RGB `/255`, and that stack order is not necessarily chronological. Its raw-S2 example illustrates EPSG:3857, 9.555 m, bilinear resampling, but does not establish those settings for these exact chips.
- `ssr/data/s2-naip_dataset.py` forms the pair by tile-key path construction. That code does not verify source coordinates or registration. `ssr/utils/infer_utils.py` selects frames from the stack and divides RGB by 255; absent a per-index mapping, a selected stack index is not a timestamp.
- `reports/satlas_sr_provenance_investigation.md`, `reports/satlas_sr_provenance_request.md`, `reports/five_tile_evaluation_audit.md`, and `reports/satlas_sr_full_validation_evaluation.md` all distinguish intended pairing from verified pixel alignment. They record no frame-to-scene/date table or per-chip affine/crop/resampling manifest in the current archive/repository.

### Consequence for the 256 tiles

| Use | Decision | Reason |
|---|---|---|
| Qualitative visualization | **Suitable with caveats** | The 256 input/output/reference files are readable, paired by intended tile key, and contact-sheet outputs exist. Visual differences are observations, not validated building error or hallucination labels. |
| Exploratory metrics | **Suitable** | Raw PSNR/SSIM were computed for all 256 same-index RGB arrays with matching shapes and remain explicitly marked exploratory. They are useful for debugging and descriptive comparison only. |
| Validated pixel-level metrics | **NO-GO** | Per-frame S2 temporal provenance, authoritative georeferencing/crop transforms, and independently checked registration/valid masks are missing. Shared tile names and dimensions cannot replace these records. |
| Reproducing paper numerical results | **NO-GO** | The local checkpoint is the small model, while the paper’s main method comparison uses medium ESRGAN and its five data-size/model-scale experiments and four-dataset comparisons are not present. The local metric protocol/results also do not match the paper’s full metric suite or split. |

The author’s paper statement that metadata is provided for all imagery makes a separate metadata source plausible, but does not prove that metadata has been located or that it covers these 256 small-validation files. Do not derive or infer precise transforms, S2 acquisition dates, or registration from tile IDs or image appearance.

## Hardware, dependencies, and resource assessment

**Observed environment (read-only snapshot):** Python 3.9.6, PyTorch 2.1.0+cu121, CUDA available, NVIDIA RTX 2000 Ada Generation, 16,380 MiB VRAM. `basicsr`, `torchvision`, `scikit-image`, `lpips`, and `open_clip` are importable. `clip` is not installed. At inspection, the GPU reported 939 MiB used and 1% utilization; this is a point-in-time state, not a capacity benchmark.

| Workload | Resource view |
|---|---|
| Existing checkpoint inference | Already completed in 23.021 seconds for 256 tiles with the current GPU, according to its run summary. This runtime does not estimate training. |
| PSNR/SSIM/LPIPS evaluation on the 256 existing outputs | Computationally modest relative to training, but scientific validity is blocked by provenance; this audit did not run it. LPIPS dependency is available, while pretrained-weight availability was not checked. |
| CLIPScore | Requires the metric’s `clip` import path to be resolved (currently absent) and the CLIPA pretrained weights to be available. The weights/cache and memory behavior were not inspected or benchmarked. |
| Small ESRGAN training | One 16-GiB GPU may support experiments only after batch-size/memory benchmarking and exact recipe confirmation. The local baseline YAML’s batch size 256 per GPU and 100M iterations should not be assumed feasible or paper-identical on this workstation. No training duration estimate is justified without measured throughput. |
| Medium/large and full scaling sweep | The 87M/347M models, five dataset fractions, and multiple methods multiply compute and data needs. A 16-GiB single-GPU machine has no demonstrated capacity for the paper schedule. Mark as unbenchmarked/high risk; do not promise feasibility or fabricate runtime. |
| Human/downstream studies | These are limited primarily by missing task datasets, labels, comparable model outputs, and human annotation/study resources rather than GPU alone. |

Local `requirements.txt` names both `git+https://github.com/openai/CLIP.git` and `open_clip_torch`, but the current environment lacks the `clip` module. No dependency installation was attempted. The unconfirmed `.crdownload` file in `experiments/` is not treated as a usable checkpoint.

## Recommended staged plan

### Tier A — Existing evidence, no new compute

- Keep `results/satlas_sr_full_validation/` and its run manifest as the record of small-checkpoint inference on the 256 local tiles.
- Use existing contact sheets for qualitative review and existing PSNR/SSIM only as exploratory raw-array statistics; preserve the explicit unverified-registration qualifier.
- Describe the achieved scope as checkpoint inference reproduction, not training or paper reproduction.

**Acceptance:** any report/article states the exact small checkpoint and 256-tile subset and labels metrics exploratory; it makes no validated pixel-quality or full-paper claim.

### Tier B — Specific author/source evidence and modest targeted work

- Obtain the official supplement (especially A.1–A.6) and clarify the intended target experiment and exact method/metric/training recipes.
- Request or locate the compact provenance manifest for the 256 existing tile pairs: ordered S2 frame index → source product/scene ID and acquisition timestamp; NAIP image/source ID and date; CRS/transforms or equivalent bounds; source and output crop windows/pixel sizes/resampling; pairing/registration procedure; and valid/cloud masks if used.
- Reconcile the paper’s two-month vs README’s three-month pairing window. Once exact split/data locations and sizes are known, decide whether to obtain a named small evaluation split or just provenance metadata; do not fetch the 974-GB public dataset repository wholesale.
- After independent spatial/temporal validation is accepted, reproduce one selected metric protocol on a pre-registered subset, retaining every tile and exclusion reason. For paper-style score reproduction, use the paper’s held-out split/model and cPSNR, LPIPS-derived agreement, and CLIPScore conventions rather than current raw PSNR/SSIM alone.
- Resolve the missing `clip` dependency path and pretrained-model availability before CLIPScore work; do not install anything until an implementation plan is approved.

**Acceptance:** manifest joins every included tile/frame unambiguously; timestamps satisfy the author-confirmed inclusion policy; both image transforms map to a stated common grid; independent residual/overlap checks meet thresholds fixed before scoring; metric code/settings and split match the selected paper protocol.

### Tier C — Large-scale reproduction or new human studies

- Training the 1/3/10/30/100% data sweep and 17M/87M/347M models requires authoritative split manifests and large training data, multiple trained/checkpointed runs, and a demonstrated compute plan. Current single-GPU throughput has not been measured.
- The four-method/four-dataset comparison requires the other benchmark datasets and method implementations/checkpoints, especially SR3, plus consistent evaluation.
- The CLIP-loss/domain-knowledge best model, DDIM/CFG experiments, downstream task tables, global deployment, expert building counts, and human preference study each require their exact supplementary recipes, additional code/data/weights, and (for human studies) labels or participants.

**Acceptance:** each claimed result has a matching paper split, model/checkpoint or documented training run, metric implementation, and complete per-example record; human-derived findings use the original labels/protocol or a separately approved study. Full reproduction is not a single checkpoint run.

## Exact next action and acceptance criteria

**Highest priority: obtain from the official authors/maintainers (or an accessible official source) the supplement and the small-validation provenance manifest as one targeted reproduction package request.** Do not treat the earlier inaccessible supplement URL as evidence that supplement details do not exist, and do not claim the authors have been contacted by this audit.

The request should ask for: (1) Supplement A.1–A.6, particularly A.3/A.4/A.5; (2) the 256 small-validation tile list and split role; (3) a per-tile/per-frame table linking each PNG stack index to S2 scene/product ID and UTC acquisition time and linking each NAIP PNG to its source image/date/footprint; (4) both image CRS/transforms or equivalent geospatial bounds, crop windows, pixel sizes and resampling/grid-origin history; (5) the exact pairing/registration and metric protocol; and (6) the authoritative small-validation archive/manifest location, format, and size.

**Pass:** supplied materials map the existing files and all selected frame indices unambiguously; specify temporal policy and spatial transforms/crop/resampling sufficiently to reconstruct both footprints on one grid; provide or cite the exact selected experiment recipe and split; and enable independent residual/overlap checks under acceptance thresholds defined before scoring.  
**Fail / remain blocked:** any frame lacks source/date identity, any chip lacks reconstructable transform/crop provenance, the materials do not identify the local 256 subset, or the supplement/recipe leaves an experiment-critical setting unspecified. In those cases, keep the current result classified as limited checkpoint inference with exploratory metrics and choose a smaller source-imagery subset with authoritative georeferencing only after its provenance is available.

This action requires author/source assistance, but **no GPU compute or dataset download** to make the first determination. If a later targeted data download is proposed, first confirm its official source, exact object path/format, expected size, and coverage. No human evaluation is needed for this provenance gate.

## Source references

### Local files inspected

- `satlas-super-resolution/README.md` — dataset path/structure, imagery bands and normalization, generic S2 reprojection example, model/checkpoint descriptions, train/test commands, and SR3 repository note.
- `satlas-super-resolution/ssr/options/experiments/esrgan_baseline.yml` — baseline data paths, batch/workers, architecture, optimizer, scheduler, loss, iteration count, and official validation/test metric settings.
- `satlas-super-resolution/ssr/options/infer_satlas_full_validation.yml` — local inference paths, 8-frame input, model definition, checkpoint and EMA key.
- `satlas-super-resolution/ssr/data/s2-naip_dataset.py`; `ssr/utils/infer_utils.py`; `ssr/infer.py` — tile-key pairing, frame loading/selection and inference preprocessing.
- `satlas-super-resolution/ssr/archs/rrdbnet_arch.py`; `ssr/archs/srcnn_arch.py`; `ssr/archs/highresnet_arch.py`; `ssr/utils/model_utils.py` — local architecture implementations and model construction.
- `satlas-super-resolution/ssr/test.py`; `ssr/metrics/cpsnr.py`; `ssr/metrics/lpips.py`; `ssr/metrics/clipscore.py`; `requirements.txt` — testing and metric implementation/dependencies.
- `satlas-super-resolution/experiments/esrgan_8S2.pth` — filename/size and run-recorded SHA-256 `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`; checkpoint was not loaded or modified during this audit.
- `satlas-super-resolution/results/satlas_sr_full_validation/run_summary.md`; `inference_manifest.csv`; `summary.json`; `tile_metrics.csv` — completed inference and exploratory metric records.
- `satlas-super-resolution/evaluate_satlas_sr_validation.py` — existing metric definitions/settings.
- `satlas-super-resolution/reports/satlas_sr_provenance_investigation.md`; `satlas_sr_provenance_request.md`; `satlas_sr_base_paper_reproduction_gap.md`; `five_tile_evaluation_audit.md`; `satlas_sr_full_validation_evaluation.md` — provenance, prior gap analysis, and evaluation limits.
- Read-only repository status was `experiments...origin/experiments`, with pre-existing modifications/untracked work. Those files were not changed.

### Official sources

- [Official paper, arXiv HTML](https://arxiv.org/html/2311.18082v1): Sections 3 (metrics and human preference), 4.1 (S2-NAIP data and scaling), 5 (method comparison and building count), 6 (CLIP loss/best model), 7 (downstream tasks), and Supplement pointer. Main paper directs detailed training/method settings to supplementary sections.
- [Official Satlas-SR repository, experiments branch](https://github.com/allenai/satlas-super-resolution/tree/experiments): dataset/model docs and published configs/checkpoint links.
- [Official Satlas-SR Hugging Face dataset listing](https://huggingface.co/datasets/allenai/satlas-super-resolution): cited by the prior local provenance investigation as the large public dataset listing; no large download was made for this audit.
- The supplementary PDF is linked from the paper and repository README. A prior read-only retrieval attempt returned HTTP 401; this audit therefore treats its detailed contents as **unresolved**, not absent.

## Final decision

**GO:** describe the existing 256-tile run as inference with the released small 8-S2 ESRGAN checkpoint; use images qualitatively and retain the current raw PSNR/SSIM as exploratory statistics.  
**NO-GO:** validated pixel-level metrics or claiming the paper’s reported numerical results/full reproduction. The checkpoint’s training lineage, exact supplement recipes, intended paper split/model, additional metrics and methods, and per-image spatial/temporal validation have not been established locally.

The smallest scientifically defensible next objective is one explicitly selected paper experiment, after the official supplement and per-image provenance are obtained and the chosen split/model/metrics are matched. No training should be proposed until that recipe and resource plan are established.
