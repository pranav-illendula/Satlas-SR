# EXP-011: Sentinel-2 Frame-Selection Sensitivity Experiment Plan

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe`  
**Hardware Target**: NVIDIA RTX 2000 Ada Generation (16 GB VRAM)  
**Status**: Proposal and Experiment Specification (Awaiting User Approval — Pure Audit Only)

---

## 1. Objective and Scientific Rationale

The Satlas-SR super-resolution model (`SSR_RRDBNet`, 4× upscaling, 24 input channels) requires exactly 8 temporal Sentinel-2 observations ($8 \times 3 = 24$ channels) to predict a high-resolution $128 \times 128$ image. 

In the official codebase:
1. The 32 available observations in each tile's `tci.png` are filtered for validity.
2. 8 frames are drawn using `random.sample(goods, 8)` with an arbitrary random seed (`seed=0`).
3. The selected frames are concatenated directly along the channel dimension.

Because:
- 2D convolutions in `conv_first` are **not permutation-invariant** across channels, and
- Different frames capture distinct atmospheric, seasonal, and lighting conditions,

this experiment investigates whether the frame selection strategy materially influences super-resolution fidelity, or whether performance is robust across selection policies.

---

## 2. Safety and Non-Regression Guarantees

To ensure complete experimental safety:
- **No Training / Fine-Tuning**: Uses exclusively the frozen official checkpoint `experiments/esrgan_8S2.pth` (SHA-256: `167d6b9d...`, EMA weights).
- **Zero Baseline Overwrite**: All new predictions, logs, and evaluation metrics will be written exclusively to dedicated subdirectories inside `results/satlas_sr_experiment_011_frame_selection_audit/evaluations/`. The baseline directory `results/satlas_sr_baseline_freeze_v1/` and previous experiment reports remain completely untouched.
- **No Data Leakage**: Ground truth NAIP images will **never** be used to select Sentinel-2 frames. All selection strategies are strictly unsupervised / blind to the target.
- **Deterministic Reproducibility**: Every strategy is fully deterministic and parameterized.

---

## 3. Candidate Frame-Selection Strategies

All candidate strategies operate on the 32 available sub-frames ($[0..31]$) of `tci.png`:

| Strategy ID | Strategy Name | Type | Selection / Ordering Rule | Scientific Question Addressed |
| :--- | :--- | :--- | :--- | :--- |
| **S0** | **Baseline Frozen (Seed 0)** | Pseudorandom | `random.seed(0)` sequential stream; exact reproduction of frozen baseline. | Control benchmark. |
| **S1** | **Alternate Seed (Seed 42)** | Pseudorandom | `random.seed(42)` sequential stream with `random.sample(goods, 8)`. | Tests pure stochastic variance across random draws. |
| **S2** | **Sorted Baseline (Permutation Isolation)** | Content-Matched | Exactly the 8 frames selected by S0, but sorted in ascending index order: `sorted(indices)`. | Isolates channel permutation effect from frame content. |
| **S3** | **Uniform Strided Grid** | Deterministic | Fixed indices: `[0, 4, 8, 12, 16, 20, 24, 28]` (step size 4 across the 32-frame buffer). | Tests uniform temporal sampling with zero randomness. |
| **S4** | **Early Window** | Deterministic | Fixed indices: `[0, 1, 2, 3, 4, 5, 6, 7]`. | Tests temporal clustering at buffer start. |
| **S5** | **Late Window** | Deterministic | Fixed indices: `[24, 25, 26, 27, 28, 29, 30, 31]`. | Tests temporal clustering at buffer end. |
| **S6** | **Low-Brightness Heuristic (Cloud Avoidance)** | Content-Heuristic | Compute mean RGB intensity for each of the 32 frames; select the 8 frames with lowest mean intensity (sorted ascending). | Tests whether automated cloud/glint avoidance improves SR fidelity without requiring metadata. |

---

## 4. Evaluation Protocol

### 4.1. Two-Tier Execution Design
1. **Tier 1 (5-Tile Smoke Pilot)**:
   - Target Tiles: Canonical pilot set (`20771_46525`, `20890_46801`, `20926_46045`, `21068_45692`, `21856_49804`).
   - Purpose: Verifies pipeline execution, GPU memory stability, and output validity in $< 5\text{ seconds}$.
2. **Tier 2 (Full 256-Tile Validation)**:
   - Target Tiles: All 256 tiles in `data/small_val_set`.
   - Purpose: Provides statistical power to assess per-tile paired differences across all strategies.

### 4.2. Metric Suite
Metrics will be computed against NAIP ground truth using the verified implementation in `evaluate_satlas_sr_paper_metrics.py`:
- **cPSNR (dB)**: Cropped PSNR (border = 4 px).
- **PSNR (dB)**: Full-frame PSNR.
- **SSIM**: Structural Similarity Index (border = 4 px).
- **LPIPS**: Perceptual distance using official VGG network (`net='vgg'`).
- **CLIPScore**: Cosine similarity via `clipa-ViT-bigG-14`.

### 4.3. Statistical Analysis Protocol
For each alternative strategy $S_k$ ($k \in \{1..6\}$):
- Compute paired per-tile difference against baseline $S_0$:
  $$\Delta M_i = M_i(S_k) - M_i(S_0) \quad \text{for tile } i \in \{1..256\}$$
- Compute:
  - Mean difference $\overline{\Delta M}$ and standard deviation $\sigma_{\Delta M}$.
  - Two-tailed Paired Student's t-test and Wilcoxon signed-rank test $p$-values.
  - 95% Confidence Interval for the mean difference.
  - Win/Loss/Tie ratio (% of tiles where $S_k$ outperformed $S_0$).

---

## 5. Expected Compute and Resource Budget

| Operation | Scale | Estimated Runtime | Peak VRAM | Disk Space |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Pilot, 7 strategies)** | $5 \text{ tiles} \times 7 = 35$ inferences | $\sim 10\text{ seconds}$ | $\sim 2.1\text{ GiB}$ | $< 2\text{ MB}$ |
| **Tier 2 Inference (Single Strategy)** | 256 tiles | $\sim 72\text{ seconds}$ | $\sim 2.1\text{ GiB}$ | $\sim 13\text{ MB}$ |
| **Tier 2 Inference (All 7 Strategies)** | $256 \times 7 = 1,792$ inferences | $\sim 8.5\text{ minutes}$ | $\sim 2.1\text{ GiB}$ | $\sim 91\text{ MB}$ |
| **Tier 2 Metric Evaluation (Per Strategy)**| 256 tiles (cPSNR, SSIM, LPIPS, CLIP) | $\sim 90\text{ seconds}$ | $\sim 3.8\text{ GiB}$ | $< 1\text{ MB}$ |
| **Tier 2 Full Metrics (All 7 Strategies)** | $256 \times 7$ evaluations | $\sim 10.5\text{ minutes}$ | $\sim 3.8\text{ GiB}$ | $< 5\text{ MB}$ |
| **Total Experiment Compute** | Full 7-strategy suite | **$\sim 19$ minutes** | **$< 4\text{ GiB}$** | **$< 100\text{ MB}$** |

All compute is completely feasible on the local NVIDIA RTX 2000 Ada GPU without swapping or risk of memory exhaustion.

---

## 6. Known Confounders and Interpretation Rules

### 6.1. Architectural Confounders
- **Channel Permutation Sensitivity**: As demonstrated in `FRAME_SELECTION_AUDIT.md`, `conv_first` applies independent 2D convolution filters to each of the 24 channels. S2 (content-matched with sorted indices) directly controls for this confounder.
- **Training Distribution Bias**: The official model was trained with random sampling. The weights may exhibit slight preference or degradation when presented with rigidly structured patterns (e.g. sorted or uniform grids).

### 6.2. Ground Truth Temporal Asynchrony
- Sentinel-2 frames span months of satellite revisit orbits, while the NAIP target represents an airborne orthophoto captured at a single specific date and time.
- A higher PSNR against NAIP does not necessarily mean "better super-resolution"; it may simply reflect an observation that was seasonally or phenologically closer to the NAIP flight date.

### 6.3. Decision and Interpretation Rules
1. **Insignificance / Robustness Threshold**:
   If $|\overline{\Delta \text{cPSNR}}| < 0.1\text{ dB}$ and $|\overline{\Delta \text{LPIPS}}| < 0.005$ with $p \ge 0.01$, conclude that Satlas-SR super-resolution quality is **robust to frame selection** within the valid 32-frame observation pool.
2. **Material Sensitivity Threshold**:
   If any strategy shows $|\overline{\Delta \text{cPSNR}}| \ge 0.25\text{ dB}$ or $|\overline{\Delta \text{LPIPS}}| \ge 0.010$ with $p < 0.001$, conclude that frame selection **materially impacts super-resolution fidelity**.
3. **Deterministic Superiority**:
   If the Uniform Strided Grid (S3) achieves comparable or superior fidelity to Baseline (S0), recommend replacing pseudorandom sampling with deterministic striding in future deployments to eliminate random seed dependencies.

---

## 7. Approval Gate

**Strict Stop Condition**: In accordance with user safety requirements, NO inference, validation runs, or metric computations will take place until the user explicitly reviews this audit and grants approval to run the proposed experiment.
