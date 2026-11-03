# Satlas-SR EXP-002: Paper Reproduction Gap Matrix

- **Experiment Identifier**: `EXP-002`
- **Date**: 2026-10-22
- **Target Paper**: *Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing* ([arXiv:2311.18082v1](https://arxiv.org/abs/2311.18082))
- **Baseline Evaluated**: Frozen Satlas-SR Baseline v1 (`results/satlas_sr_baseline_freeze_v1/`, `experiments/esrgan_8S2.pth`, `data/small_val_set`)
- **Metric Diagnostic**: EXP-001 Dense cPSNR Sensitivity Analysis (`results/satlas_sr_experiment_001/`)
- **Output Artifact**: `results/satlas_sr_experiment_002_paper_gap_analysis/PAPER_REPRODUCTION_GAP_MATRIX.md`

---

## 1. Executive Summary & Purpose

This audit provides a rigorous, evidence-based reconciliation between the published Satlas-SR paper (*Wolters et al., arXiv:2311.18082v1*) and the local repository baseline. 

The primary finding of this gap analysis is that **the local baseline is an exact, verified reproduction of the released small ESRGAN checkpoint (`esrgan_8S2.pth`, 16.71M parameters) on the small urban validation subset (`small_val_set`, 256 tiles)**. However, **it is not a reproduction of the primary published cross-method benchmark (Table 2)**, which evaluated an **87M-parameter medium ESRGAN** across nationwide multi-biome test splits.

Below is the itemized 10-dimension comparison matrix classifying all aspects of model architecture, data, training, preprocessing, and evaluation.

---

## 2. Paper Reproduction Gap Matrix

| # | Dimension | Classification | Stated Paper Specification | Local Implementation & Inspected Evidence | Source File & Config Keys | Scientific Gap & Required Next Action |
| :-: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | **Model Variant & Parameter Count** | **MISMATCH** (vs. Table 2) / **PARTIAL** (vs. §4.1) | Paper Section 4.1 defines three ESRGAN variants: Small (~17M), Medium (~87M), Large (~347M). Table 2 uses the **87M Medium** variant to benchmark against 97M SR3. | Local checkpoint `experiments/esrgan_8S2.pth` contains exactly **16,710,083 parameters** (~16.71M, Small). The Medium generator config exists (`esrgan_larger1.yml`, 66.8M generator + 17.5M discriminator = 84.3M), but no weights exist locally. | Checkpoint: `experiments/esrgan_8S2.pth`<br>Config: `ssr/options/experiments/esrgan_baseline.yml`<br>Medium Config: `ssr/options/experiments/esrgan_larger1.yml` | The small checkpoint cannot be directly compared against Table 2's 87M model. Need author-released medium weights or full retraining to match Table 2. |
| **2** | **Model Architecture & Checkpoint Identity** | **PARTIAL** | Multi-frame ESRGAN generator (`SSR_RRDBNet`), 23 RRDB blocks, scale 4. Model checkpoint trained on 100% S2-NAIP with 8 Sentinel-2 frames. | Architecture code matches exactly. Instantiated `SSR_RRDBNet` loads `experiments/esrgan_8S2.pth` (`params_ema`) with 100% strict matching (zero missing/unexpected keys). Checkpoint SHA-256: `167d6b9d...`. | Code: `ssr/archs/rrdbnet_arch.py`<br>Config: `ssr/options/infer_satlas_full_validation.yml`<br>Key: `network_g: type: SSR_RRDBNet` | Exact match for small model inference. However, no training logs, optimizer states, or medium/large checkpoint weights are present. |
| **3** | **S2 Input Frames & Frame Selection** | **PARTIAL** | 8 Sentinel-2 frames per NAIP chip ($24$ channels). Paper states: *"We do not remove images with cloud cover, so that the data maintains a real-world distribution."* Stack contains $\ge 18$ frames within 2 months of NAIP. | Local pipeline takes stacked $(1024, 32, 3)$ PNG (up to 32 frames), filters out frames containing black pixels (`[0,0,0]`), and uniformly samples 8 frames using `random.sample`. Evaluated under `manual_seed: 0`. | Script: `ssr/utils/infer_utils.py`<br>Function: `format_s2naip_data`<br>Config: `ssr/options/infer_satlas_full_validation.yml`<br>Key: `n_s2_images: 8` | Frame count (8) matches. However, the exact per-tile frame timestamps/scene IDs selected during paper benchmark evaluation were not published. |
| **4** | **Image Dimensions, Scale, Normalization, & Preprocessing** | **MATCH** | S2 input: $32 \times 32$ per frame ($24 \times 32 \times 32$ total), normalized via $\text{RGB} / 255.0 \in [0, 1]$. Target: NAIP $128 \times 128$ RGB (downsampled $4\times$ from $512 \times 512$). Scale factor 4. | Exact input tensor shape `(1, 24, 32, 32)`, values in $[0.0, 1.0]$. Exact output tensor shape `(1, 3, 128, 128)`, clamped to $[0, 1]$, quantized to uint8 $\times 255$. | Script: `ssr/utils/infer_utils.py`<br>Script: `run_full_validation_inference.py`<br>Keys: `scale: 4`, `num_in_ch: 24`, `num_out_ch: 3` | Fully identical. Preprocessing and tensor pipelines match paper specifications completely. |
| **5** | **Training, Validation, & Test Datasets** | **MISMATCH** | S2-NAIP dataset: 1.2M training pairs (1%, 3%, 10%, 30%, 100% splits). Validation set: 8,192 pairs. Test set: eight $16 \times 16$ grids (2,048 tiles) from diverse global regions, plus held-out test splits. | Local dataset contains only `data/small_val_set` (256 urban pairs). No training data (`train_urban_set` ~1.1M pairs or `train_full_set` ~44M pairs) is downloaded. Full `val_set` (8,192 pairs) is not downloaded. | Directory: `data/small_val_set/`<br>README: `README.md` (lines 41–67) | Local evaluation is strictly limited to 256 urban tiles (3.125% of `val_set`). Evaluating full benchmark requires downloading `val_set` (~40 GB). |
| **6** | **Geographic Coverage & Biome Diversity** | **MISMATCH** | Continental US coverage within 20km of cities $\ge 20\text{k}$ pop; includes farmland, forest, mountains, water, and residential. Global test sets cover Bolivia, France, South Africa, Japan. | `small_val_set` consists exclusively of 256 urban tiles from US cities. Rural, desert, dense forest, and international tiles are absent. | Manifest: `results/satlas_sr_baseline_freeze_v1/baseline_manifest.json`<br>Data: `data/small_val_set/naip/` | High urban structural density skews metric distributions (e.g. higher CLIPScore, lower raw PSNR) compared to nationwide mixed-biome benchmarks. |
| **7** | **Loss Functions & Training Configuration** | **PARTIAL** | ESRGAN trained with L1 pixel loss, VGG19 perceptual loss, and Vanilla/Relativistic GAN loss. Adam optimizer ($10^{-4}$), MultiStepLR. Aux CLIPScore loss in Section 6. | Config `esrgan_baseline.yml` specifies exact paper loss formulations: L1 (1.0), VGG19 (1.0), GAN (0.1), Adam ($10^{-4}$). `cliploss` config exists. However, no training was conducted locally; no training logs/resumes exist. | Config: `ssr/options/experiments/esrgan_baseline.yml`<br>Keys: `pixel_opt`, `perceptual_opt`, `gan_opt`, `optim_g` | Architecture and loss definitions match published code, but empirical training dynamics and convergence endpoints remain unverified locally. |
| **8** | **Evaluation Metrics & Crop / Search Settings** | **MATCH** (cPSNR/CLIP) / **PARTIAL** (LPIPS) | cPSNR: PROBA-V translational search (window 8, crop border 4). CLIPScore: OpenCLIP `ViT-bigG-14-CLIPA-336` (DataComp1b). LPIPS: VGG trunk. | `ssr/metrics/cpsnr.py` implements exact PROBA-V cPSNR (81 offsets, stride 2, border 4, per-channel bias). `ssr/metrics/clipscore.py` uses exact CLIPA model. `ssr/metrics/lpips.py` uses VGG. | Code: `ssr/metrics/cpsnr.py`<br>Code: `ssr/metrics/clipscore.py`<br>Code: `ssr/metrics/lpips.py` | Metric code in repo matches paper design. However, paper Table 2 reports LPIPS as human agreement % ($\uparrow$), while code outputs raw distance ($\downarrow$). |
| **9** | **Reported Metric Values & Numerical Comparison** | **PARTIAL** / **MISMATCH** | Table 2 (S2-NAIP Medium ESRGAN):<br>• cPSNR: **22.6506 dB**<br>• LPIPS: **0.8406** ($\uparrow$, agreement %)<br>• CLIPScore: **0.8745**<br>Section 6 Best Model: CLIPScore: **0.959** | Local Small ESRGAN (256 urban tiles):<br>• cPSNR (stride 2): **20.21 ± 4.15 dB**<br>• Standard PSNR: **18.89 ± 3.86 dB**<br>• LPIPS: **0.347 ± 0.081** ($\downarrow$, raw distance)<br>• CLIPScore: **0.961 ± 0.034** | Paper: Table 2 ([arXiv:2311.18082v1](https://arxiv.org/html/2311.18082v1#S4.T2))<br>Baseline: `results/satlas_sr_baseline_freeze_v1/BASELINE_FREEZE_REPORT.md` | Values cannot be equated: local cPSNR is 2.44 dB lower due to small model capacity (17M vs 87M) and urban tile complexity; CLIPScore is higher due to urban feature density. |
| **10** | **Random Seeds, Inference Settings, & Determinism** | **PARTIAL** | Paper does not publish global random seed or per-tile frame selection manifests for the published test tables. | Local pipeline enforces deterministic tile sorting (`sorted(s2_dirs)`), sets `manual_seed: 0`, and records all 256 selected frame indices atomically in `inference_manifest.csv`. | Script: `run_full_validation_inference.py`<br>Manifest: `results/satlas_sr_full_validation/inference_manifest.csv` | Local inference is 100% reproducible and auditable. Paper's exact frame sampling sequence across its 1.2M dataset remains **UNKNOWN**. |

---

## 3. Deep-Dive Reconciliations

### 3.1 Model Variant Reconciliation (Small 17M vs. Medium 87M)
- **Empirical Ground Truth**: Checkpoint `experiments/esrgan_8S2.pth` contains **16,710,083 parameters** (`SSR_RRDBNet(num_in_ch=24, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)`).
- **Paper Context**: Section 4.1 explicitly outlines three ESRGAN capacities:
  1. **Small**: ~17M parameters (matches `experiments/esrgan_8S2.pth`).
  2. **Medium**: ~87M parameters (`esrgan_larger1.yml`: `num_feat: 128`, `num_grow_ch: 64`, generator = 66.8M, discriminator = 17.5M, total = 84.3M).
  3. **Large**: ~347M parameters (`esrgan_larger2.yml`: `num_feat: 256`, `num_grow_ch: 128`, generator = 267.0M, discriminator = 70.1M, total = 337.1M).
- **The Table 2 Discrepancy**: In Section 5, the authors explain:
  > *"The SR3 model has 97mil parameters so we choose to use the medium ESRGAN from Figure 5 which has 87mil parameters, for a relatively fair comparison."*
- **Conclusion**: The local baseline is the **Small (17M) model**, which corresponds to the scaling studies in Section 4.1 (Figure 5), **not the Medium (87M) model reported in Table 2**.

### 3.2 Metric Comparability & Protocol Discrepancies
1. **cPSNR Search Stride (EXP-001 Finding)**:
   - Official implementation (`ssr/metrics/cpsnr.py`) evaluates 81 offsets with `stride=2`.
   - EXP-001 proved that a dense `stride=1` search evaluates 289 candidates, yielding an empirical mean gain of **+0.1232 dB** (95% CI: $[+0.1018, +0.1477]\text{ dB}$).
   - **Protocol Rule**: Stride-2 cPSNR must remain the primary reported metric to preserve fidelity with published paper code. Dense cPSNR is an evaluation diagnostic, not a model improvement.
2. **LPIPS Distance vs. Preference Agreement**:
   - Local script outputs raw LPIPS distance ($\downarrow$, where $0$ is identical; baseline mean = $0.347$).
   - Paper Table 2 reports LPIPS as human preference agreement percentage ($\uparrow$, where $1.0$ is perfect; ESRGAN = $0.8406$). Comparing $0.347$ to $0.8406$ directly is invalid without the conversion formula from Supplementary Section A.1.
3. **CLIPScore Range & Urban Subset Bias**:
   - Baseline CLIPScore on 256 urban tiles is **$0.961 \pm 0.034$**, whereas Table 2 reports **$0.8745$** on the full S2-NAIP benchmark.
   - The higher score on `small_val_set` is driven by dense geometric patterns in urban areas (buildings, roads, rectilinear footprints) which align strongly with foundation vision features, whereas the nationwide benchmark includes low-texture desert, forest, and agricultural biomes.

---

## 4. Classification Summary

- **MATCH (2/10)**:
  - Input/Target dimensions, scale factor 4, and normalization ($/255.0$).
  - cPSNR / CLIPScore repository implementations.
- **PARTIAL (5/10)**:
  - Architecture code and small checkpoint identity.
  - S2 frame count (8 frames) and sampling mechanics.
  - Loss functions and training configurations.
  - Metric definitions and output conversions (LPIPS agreement vs. raw distance).
  - Reproducibility settings and determinism (local is deterministic; paper seed unknown).
- **MISMATCH (3/10)**:
  - Model variant and parameter count for Table 2 (16.71M Small vs. 87M Medium).
  - Evaluation dataset split (256-tile urban subset vs. full nationwide benchmark).
  - Geographic coverage and biome diversity (US urban subset vs. multi-biome global data).

---

## 5. Provenance & Audit Trail
- Evaluated by: Senior Research Engineer
- Checkpoint Hash: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`
- Paper Version: arXiv:2311.18082v1 (November 2023)
