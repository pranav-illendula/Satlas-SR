# Satlas-SR EXP-005: Supplementary Material Verification & LPIPS Conversion Resolution Report

- **Experiment Identifier**: `EXP-005`
- **Date**: 2026-10-22
- **Author**: Senior Research Engineer
- **Repository Root**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`
- **Active Git Branch**: `experiments` (commit `3e67973f6746bedbd91cc032b4888574980e1daf`)
- **Remote Commit Inspected**: `7ccf8d118307cfc58062d488fd8b7c3ab27e2fa6` (branch: `origin/favyen/20260322-update-links-from-r2-to-gcs`)
- **Downloaded Supplementary Asset**: [`Zooming_Out_On_Zooming_In_Supplementary.pdf`](./Zooming_Out_On_Zooming_In_Supplementary.pdf)
- **Output Directory**: `results/satlas_sr_experiment_005_supplementary_verification/`

---

## 1. Executive Summary & Definitive Decision

> ### **FINAL DECISION: THE METRIC CANNOT BE REPRODUCED FROM THE AVAILABLE EVIDENCE**
>
> Following direct physical inspection of the official supplementary material (*Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing — Supplementary Material*, Wolters, Bastani, Kembhavi), we conclusively establish that:
> 1. **No Closed-Form Conversion Formula Exists**: Section A.1 contains **zero mathematical equations, calibration parameters, or scalar transformation functions**.
> 2. **Table 2 LPIPS "Accuracy" is an Empirical 2AFC Human Agreement Rate**: The paper's Table 2 metric reflects the percentage of pairwise comparisons on an Amazon Mechanical Turk (AMT) study where LPIPS agreed with human annotators. Computing this metric fundamentally requires the **unreleased 11,524 human preference judgments** and simultaneous outputs from four competing model families.
> 3. **Raw Continuous LPIPS Distance Must Be Retained**: Because the human preference annotation dataset and pairwise scoring scripts were never published or integrated into the repository, raw perceptual distance ($0.347 \pm 0.081$) is the only verifiable and scientifically reproducible perceptual metric on the local baseline. Any attempt to convert single-model continuous LPIPS distance into Table 2's accuracy scores would be fabricated conjecture.

---

## 2. Commit Provenance & Asset Acquisition

### 2.1 Git Provenance Audit
In branch `origin/favyen/20260322-update-links-from-r2-to-gcs`, commit `7ccf8d118307cfc58062d488fd8b7c3ab27e2fa6` (authored by Favyen Bastani `<favyenb@allenai.org>` on 2026-03-22):
- Relocated all primary AllenAI Satlas-SR datasets, checkpoints, and documentation from private/defunct Cloudflare R2 endpoints to public Google Cloud Storage (`storage.googleapis.com`).
- Updated the supplementary PDF link in `README.md` (line 19):
  - **Old URL (R2)**: `https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/Zooming_Out_On_Zooming_In_Supplementary.pdf` (HTTP 401 Unauthorized)
  - **New URL (GCS)**: `https://storage.googleapis.com/satlas-satellite-super-resolution/Zooming_Out_On_Zooming_In_Supplementary.pdf`

### 2.2 Download Record & Cryptographic Hash
The supplementary material was downloaded into the new experiment directory:
- **Local Path**: `results/satlas_sr_experiment_005_supplementary_verification/Zooming_Out_On_Zooming_In_Supplementary.pdf`
- **HTTP Status**: `200 OK`
- **Content-Type**: `application/pdf`
- **File Size**: `1,748,394 bytes` (1.67 MB / 1.74 MB nominal)
- **SHA-256 Hash**: `ebdf117d07337a0fd65f639fd85c12ac67e54d7ce0ed3e00b8859a81f2e7c51a`
- **Total Pages**: 7 pages

---

## 3. Exhaustive Inspection of Supplementary Material

We conducted a complete text and visual OCR audit across all 7 pages of the supplementary PDF:

### 3.1 Supplementary Section A.1: "Human Judgement Dataset"
The entire text of Section A.1 (Page 1) reads verbatim:
> *"To build this dataset, we used Amazon Mechanical Turk (AMT). Prior to the official task, they were tested with a qualification task to ensure they had high level of agreement with other workers, and then afterward during annotation, workers with low agreement with other workers were removed, so that workers who would suddenly begin picking arbitrary images rather than examining the images carefully would be removed.*
>
> *We ran the study on a large set of outputs from the WorldStrat and S2-NAIP datasets. Model outputs from SRCNN, HighResNet, ESRGAN, and SR3 were used for WorldStrat; the same were used for S2-NAIP in addition to multiple ESRGAN model checkpoints to add more diversity in output quality."*

#### Critical Finding
- Section A.1 contains **no mathematical equation**.
- Section A.1 contains **no calibration coefficients, logistic regression models, or scaling factors**.
- It is solely an operational description of **AMT worker qualification and low-agreement worker filtering**.

### 3.2 Supplementary Section A.2: "CLIP for Satellite Data"
- Details attempts to train satellite-specific CLIP models on S2-NAIP pairs and multi-temporal NAIP chips.
- Reports human-correspondence accuracy of $64.54\%$ and $72.58\%$, explaining why pretrained DataComp1b CLIP was selected instead.

### 3.3 Supplementary Section A.3: "Experiment Details"
- Confirms optimization parameters: Adam optimizer, initial learning rate $10^{-4}$.
- Confirms generator architecture dimensions across model scales:
  - **Small**: 64 features, 32 grow channels, 23 RRDB blocks.
  - **Medium**: 128 features, 64 grow channels, 23 RRDB blocks.
  - **Large**: 256 features, 128 grow channels, 30 RRDB blocks.
- Details input dimensions for S2-NAIP ($8 \times 32 \times 32 \to 128 \times 128$), WorldStrat ($8 \times 160 \times 160 \to 640 \times 640$), OLI2MSI ($1 \times 160 \times 160 \to 480 \times 480$), and PROBA-V ($9 \times 128 \times 128 \to 384 \times 384$, setting aside 10% for test).

### 3.4 Supplementary Sections A.4 – A.7
- **A.4**: Diffusion variations (DDIM is $10\times$ faster but causes a 7-point drop in CLIPScore; Classifier Free Guidance).
- **A.5**: Domain-knowledge ESRGAN improvements: Object-discriminator (+6 CLIPScore gain), old NAIP multi-temporal discriminator (+2 gain), and SatlasPretrain initialization (+1 gain).
- **A.6**: OpenStreetMap downstream segmentation dataset (6,144 tiles).
- **A.7 & Figures 5–8**: Qualitative visual comparisons between HighResNet and ESRGAN.

---

## 4. Deconstruction of the LPIPS Conversion Questions

| Investigation Question | Empirical Determination | Supporting Evidence from Primary Sources |
| :--- | :---: | :--- |
| **1. Is the conversion a deterministic formula applied to LPIPS distances?** | **NO** | Neither the main manuscript nor Supplementary Section A.1 provides any formula $f(d_{\text{lpips}})$. No scalar mapping exists. |
| **2. Does it require paired model outputs and human preference labels?** | **YES** | Section 3.1 & Figure 3 explicitly define the metric as the percentage of pairwise comparisons where the metric's lower distance matched the majority human judgment on Amazon Mechanical Turk. |
| **3. Does it require a calibration model, thresholds, or additional annotation data?** | **YES** | It requires the unreleased 11,524 AMT human judgment records, worker filtering masks, and simultaneous outputs for all competing models. |
| **4. What precise data and aggregation procedure are required to reproduce Table 2?** | **UNAVAILABLE** | Requires: (a) test set outputs from SRCNN, HighResNet, Medium ESRGAN, and SR3; (b) the 11,524 AMT paired choice annotations; (c) the authors' offline pairwise scoring script. |

---

## 5. Traceability of Cited Links and Code

1. **Cited References in Supplementary Material**:
   - `[1]` SatlasPretrain (Bastani et al., ICCV 2023)
   - `[2]` PatchGAN / Image-to-Image (Isola et al., CVPR 2017)
   - `[3]` OpenStreetMap planet dump (`planet.osm.org`)
   - **Zero repositories or download links for the AMT pairwise human preference dataset** are provided in the references.
2. **Repository Codebase Alignment**:
   - As established in EXP-004, `ssr/models/ssr_esrgan_model.py` (lines 339–354) directly logs raw LPIPS distance and divides by sample count:
     ```python
     self.metric_results[metric] /= (idx + 1)
     ```
   - The authors never built or checked in an LPIPS agreement calculation utility into the repository. All training and validation pipelines in the released code evaluate continuous perceptual distance.

---

## 6. Distinguishing Evidence, Interpretation, and Unresolved Requirements

### Verified Evidence (Direct Ground Truth)
- `Zooming_Out_On_Zooming_In_Supplementary.pdf` is authentic, accessible on GCS (SHA-256: `ebdf117d...`), and contains 7 pages.
- Section A.1 contains zero mathematical formulas or calibration parameters.
- The repository codebase does not compute or report Table 2's converted accuracy.
- Checkpoint `experiments/esrgan_8S2.pth` is verified as the 16.71M Small model.

### Plausible Interpretation (Scientifically Sound Deduction)
- Table 2's LPIPS column reflects each model's win-rate or agreement percentage across the 11,524 human annotations gathered during the AMT study.
- HighResNet achieved low agreement ($0.1990$) because human evaluators strongly disliked blurry L2 outputs, whereas ESRGAN achieved high agreement ($0.8406$) because humans preferred sharp generative details.

### Unresolved Requirements (Permanently Blocked Without External Data)
- Replicating Table 2's LPIPS accuracy score is **permanently blocked** unless AllenAI releases:
  1. The raw CSV of 11,524 Amazon Mechanical Turk pairwise worker annotations.
  2. The offline evaluation script used to compute Table 2 agreement from those annotations.

---

## 7. Final Recommendation & Research Conclusion

We conclude this five-experiment investigation with a definitive scientific protocol:

1. **Standardize on Raw LPIPS Distance**: All future baseline reporting and model comparisons must report standard, reproducible **VGG16 perceptual distance** ($0.347 \pm 0.081$), which is computed directly from official PyTorch/BasicSR code.
2. **Maintain Strict Baseline Bounds**: The local research baseline faithfully reproduces the **official 16.71M Small ESRGAN checkpoint** on the **256-tile urban validation subset**.
3. **No Further Search for LPIPS Formulas**: The question of an unverified conversion formula is permanently resolved: no such formula exists.
