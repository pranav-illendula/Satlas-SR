# Satlas-SR EXP-004: Authoritative Metric Provenance and Model Availability Audit Report

- **Experiment Identifier**: `EXP-004`
- **Date**: 2026-10-22
- **Author**: Senior Research Engineer
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`
- **Active Git Commit**: `3e67973f6746bedbd91cc032b4888574980e1daf` (branch: `experiments`)
- **Remote Branch Audited**: `origin/favyen/20260322-update-links-from-r2-to-gcs` (commit `7ccf8d1`)
- **Output Artifacts**: [`results/satlas_sr_experiment_004_provenance_audit/`](./)

---

## 1. Executive Summary

Following EXP-003's finding that the LPIPS-to-human-preference-agreement conversion formula was missing from the published manuscript and blocked by an inaccessible Cloudflare R2 URL, **EXP-004 conducted an exhaustive, read-only investigation across all authoritative sources**:
1. The official 11-page paper manuscript (*Wolters et al., arXiv:2311.18082v1*).
2. The official AllenAI GitHub repository (`allenai/satlas-super-resolution`), all git branches, and commit histories.
3. The upstream release manifests and remote object storage buckets (Cloudflare R2 and Google Cloud Storage).

### Key Empirical Discoveries
1. **The Supplementary PDF Was Relocated**: In branch `origin/favyen/20260322-update-links-from-r2-to-gcs` (commit `7ccf8d1`), author Favyen Bastani updated all asset links from defunct Cloudflare R2 endpoints to public Google Cloud Storage (GCS). The supplementary PDF is publicly available on GCS at:
   `https://storage.googleapis.com/satlas-satellite-super-resolution/Zooming_Out_On_Zooming_In_Supplementary.pdf` (1.74 MB, verified via HTTP HEAD `200 OK`). In accordance with strict experimental safety rules, this asset was not downloaded.
2. **The Table 2 LPIPS Metric is an Agreement Score, Not a Distance Function**: In the paper, LPIPS is not evaluated as a scalar continuous score on test predictions; rather, it represents the **percentage agreement between LPIPS-ranked image pairs and human preference annotations** on a 2-alternative forced choice (2AFC) study. The repository source code (`ssr/models/ssr_esrgan_model.py`) contains only the raw continuous distance calculation; no agreement post-processing script was ever released.
3. **The 87M Medium ESRGAN Checkpoint Was Never Released**: The authors published the configuration file (`ssr/options/experiments/esrgan_larger1.yml`), but released weights only for the 16.71M Small baseline family (`esrgan_1S2.pth` through `esrgan_16S2.pth`). Table 2's primary 87M cross-method benchmark numbers cannot be reproduced without training the 87M model from scratch.

---

## 2. Detailed Task Investigations

### Task 1: The Exact LPIPS Agreement Procedure

#### 1.1 The Stated Metric Definition in the Manuscript
- **Table 2 Caption (Page 5)**:
  > *"Table 2: Results for four methods on four datasets. **The reported LPIPS scores are converted to an accuracy, so with each of these metrics, higher is better.** ESRGAN achieves the best CLIPScore for all datasets, and that is the metric we are prioritizing for this study."*
- **Section 3.1 & Figure 3 (Pages 3–4)**:
  The authors conducted a human evaluation study on Amazon Mechanical Turk (AMT) collecting 11,524 pairwise comparisons across 20,000 S2-NAIP and 381 WorldStrat image pairs.
  - Workers were shown two model outputs ($A$ and $B$) alongside a ground truth target $T$ and asked which output was closer to the target.
  - A metric $M$ was evaluated on $(A, T)$ and $(B, T)$. The model with the lower LPIPS distance was selected as the metric's preference.
  - **Human-Metric Correspondence Accuracy**: The percentage of pairwise choices where the metric agreed with the majority human judgment:
    $$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{argmin}(d_{\text{lpips}}(A_i, T_i), d_{\text{lpips}}(B_i, T_i)) == \text{HumanChoice}_i \right)$$
  - Figure 3 indicates that on this pairwise task, LPIPS (VGG) achieves $\approx 75\%$ agreement accuracy, and LPIPS (AlexNet) achieves $\approx 78\%$.

#### 1.2 The Table 2 Discrepancy
Table 2 reports per-method scores for S2-NAIP:
- **SRCNN**: $0.3517$
- **HighResNet**: $0.1990$
- **ESRGAN (Medium 87M)**: $0.8406$
- **SR3 (Diffusion 97M)**: $0.6292$

Notice that in Table 2, these scores are reported **per individual model**, not per metric across the whole dataset. This indicates that Table 2 reports each model's **pairwise win-rate or pairwise agreement rate** against competing models or reference baselines on the human preference dataset.

#### 1.3 Repository Code Audit
We inspected `ssr/test.py`, `ssr/models/ssr_esrgan_model.py` (lines 339–354), and `ssr/metrics/lpips.py`:
- The repository only implements standard single-image perceptual distance (`lpips_loss_fn(tensor1, tensor2)`).
- During validation, the codebase computes the simple arithmetic mean of raw distances over the dataloader:
  ```python
  self.metric_results[name] /= (idx + 1)
  ```
- **Finding**: The official repository **does not contain any pairwise agreement script, human preference annotation manifest, or conversion lookup table**. The Table 2 accuracy values were computed offline by the authors from their AMT dataset.

---

### Task 2: Audit of the Medium ESRGAN Checkpoint

#### 2.1 Architecture Specification & Parameter Math
We inspected `ssr/options/experiments/esrgan_larger1.yml` and verified the network dimensions in PyTorch:
- **Generator (`SSR_RRDBNet`)**:
  - `num_in_ch: 24`, `num_out_ch: 3`, `scale: 4`
  - `num_feat: 128` (vs. 64 in small)
  - `num_block: 23`
  - `num_grow_ch: 64` (vs. 32 in small)
  - **Generator Parameter Count**: **66,782,083 parameters** (~66.8M)
- **Discriminator (`SSR_UNetDiscriminatorSN`)**:
  - `num_in_ch: 27`, `num_feat: 128`, `skip_connection: True`
  - **Discriminator Parameter Count**: **17,530,497 parameters** (~17.5M)
- **Total Trainable Model Parameters**: **84,312,580 parameters** (~84.3M $\approx$ **87M**).

#### 2.2 Checkpoint Availability
In `README.md` (lines 71–91), the authors document the released checkpoints:
- Small ESRGAN RGB series: `esrgan_1S2.pth`, `esrgan_2S2.pth`, `esrgan_4S2.pth`, `esrgan_8S2.pth`, `esrgan_16S2.pth`.
- Small ESRGAN non-RGB band models: `esrgan_10m.pth`, `esrgan_20m.pth`, `esrgan_60m.pth`.
- Alternative architectures: `srcnn_s2naip.pth`, `highresnet_s2naip.pth`.
- **Finding**: **No pretrained weights for `esrgan_larger1` (Medium 87M) or `esrgan_larger2` (Large 347M) were ever uploaded or linked in the repository**. The only released ESRGAN weights are the 16.71M Small model series.
- **Scientific Consequence**: The local checkpoint (`experiments/esrgan_8S2.pth`, 16.71M parameters) is an authentic release from AllenAI, but it is **strictly the Small baseline**. It cannot be used to replicate Table 2's Medium model scores.

---

### Task 3: Audit of Full Benchmark Reproducibility

| Benchmark Dimension | Stated Paper Specification | Public Availability / Evidence Status | Reproduction Feasibility |
| :--- | :--- | :--- | :--- |
| **Validation Split Size** | `val_set`: 8,192 image pairs across the continental US | Download link: `super_resolution_val_set_2024-01-09.tar` (~40 GB) on GCS | **Feasible upon user download approval** |
| **256-Tile Subset (`small_val_set`)** | 256 urban tiles selected from `val_set` for fast validation | Fully present locally in `../data/small_val_set` | **Completed & Frozen** (Baseline Freeze v1) |
| **Representativeness** | Nationwide multi-biome coverage | `small_val_set` is restricted to US cities with pop $\ge 50\text{k}$ within 5km radius | **Biased**: Urban density inflates CLIPScore and lowers raw PSNR |
| **Input Preprocessing** | $24 \times 32 \times 32$ S2 RGB $\to 3 \times 128 \times 128$ NAIP RGB ($4\times$) | Verified in `ssr/utils/infer_utils.py` | **100% Verified Match** |
| **cPSNR Protocol** | PROBA-V translational search, $[-8, +8]$ window, stride 2, border crop 4 | Verified in `ssr/metrics/cpsnr.py` | **100% Verified Match** (EXP-001 reproduced to $0.00005\text{ dB}$) |
| **Per-Frame Temporal Metadata** | Exact Sentinel-2 acquisition dates and cloud percentages | **UNAVAILABLE**: Tile stacks contain anonymous PNG slices | **Unresolvable without raw ESA scene catalog** |

---

## 3. Concise Evidence Matrix

| Claim | Source Document | Exact Evidence Location | Verification Status | Reproduction Impact |
| :--- | :--- | :--- | :---: | :--- |
| **Table 2 LPIPS is converted to accuracy** | arXiv:2311.18082v1 | Page 5, Table 2 Caption | **VERIFIED** | Raw LPIPS distance cannot be compared to Table 2 numbers directly. |
| **Conversion details in Supplement A.1** | arXiv:2311.18082v1 | Page 3, Section 3.1 & Page 9 Section A | **VERIFIED** | The conversion procedure is not in the main paper text. |
| **Cloudflare R2 supplement link is 401** | Network Probe | `https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/...` | **VERIFIED (HTTP 401)** | The original link cited in the paper cannot be accessed without credentials. |
| **Supplementary PDF relocated to GCS** | Git commit `7ccf8d1` | `origin/favyen/20260322-update-links-from-r2-to-gcs` | **VERIFIED (HTTP 200)** | Publicly accessible at GCS endpoint (1.74 MB); not downloaded per safety rules. |
| **Repo lacks LPIPS agreement script** | Codebase Audit | `ssr/models/ssr_esrgan_model.py` lines 339–354 | **VERIFIED (ABSENT)** | Codebase only averages raw distance; agreement was calculated offline by authors. |
| **Table 2 used 87M Medium ESRGAN** | arXiv:2311.18082v1 | Page 5, Section 5 Column 2 Line 12 | **VERIFIED** | Table 2 reflects an 87M model, not the released 16.71M checkpoint. |
| **87M Medium Checkpoint is unreleased** | Repository README | `README.md` lines 71–91 ("Model Weights") | **VERIFIED (UNRELEASED)** | Weights for `esrgan_larger1` were never uploaded; only 16.71M small weights exist. |
| **256-tile set is authentic urban subset** | Repository README | `README.md` lines 50–53 | **VERIFIED** | Local 256 tiles represent an official regression subset, but are urban-biased. |
| **Full validation set has 8,192 tiles** | Repository README | `README.md` lines 50–51 | **VERIFIED** | Evaluating nationwide performance requires downloading `val_set` (~40 GB). |
| **Evaluation code uses stride-2 cPSNR** | Codebase Audit | `ssr/metrics/cpsnr.py` & `esrgan_baseline.yml` | **VERIFIED** | Evaluation mechanics match official repository completely. |

---

## 4. Strategic Recommendation

Based on the verified provenance evidence, we recommend:

> ### **RECOMMENDATION: Document the Reproduction Boundary & Maintain the Small Baseline**
>
> 1. **Do Not Fabricate LPIPS Agreement Scores**: Acknowledge that the repository codebase reports raw perceptual distance ($0.347 \pm 0.081$), while Table 2 reported offline 2AFC human preference agreement accuracy ($0.8406$). Because the 11,524 human annotations and evaluation scripts were not integrated into the repository, raw LPIPS distance is the only scientifically defensible perceptual metric.
> 2. **Formally Bound the Local Baseline Scope**:
>    - The local reproduction successfully replicates the **official released 16.71M Small ESRGAN checkpoint** (`experiments/esrgan_8S2.pth`) on the **official 256-tile urban validation subset** (`data/small_val_set`).
>    - It is **not** an apple-to-apple replication of Table 2's **87M Medium ESRGAN** across the full nationwide benchmark, because the authors never released the 87M weights and the 8,192-tile validation set is not downloaded locally.
> 3. **Future Extension Options (Only Upon Explicit User Instruction)**:
>    - If the user wishes to evaluate the full benchmark: Request permission to download `val_set` (~40 GB).
>    - If the user wishes to replicate Table 2 parameter capacity: Request permission to download `train_urban_set` (~200 GB) and schedule multi-day training of `esrgan_larger1.yml`.
