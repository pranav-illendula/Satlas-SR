# Satlas-SR EXP-006: Final Baseline Reproducibility and Reporting Audit

**Audit Status**: Concluded & Fully Verified  
**Audit Date**: October 22, 2026  
**Auditor**: Antigravity Autonomous Coding & Verification System  
**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Git Branch**: `master` (Commit `eb3dcae4cae4492652b04c86cb31d167f8b9d997`)  
**Hardware & Software Environment**: Windows 11 Pro, Python 3.9.6, PyTorch 2.1.0+cu121, CUDA 12.1, NVIDIA RTX 2000 Ada Generation (16 GB VRAM)

---

## 1. Executive Summary

This audit report represents the definitive, evidence-based synthesis of the Satlas-SR super-resolution reproduction project (*Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing*, Wolters et al., arXiv:2311.18082v1). 

Following the completion of the baseline freeze (`v1`) and sequential controlled investigations (**EXP-001** through **EXP-005**), this read-only audit consolidates all empirical findings, resolves prior discrepancies, verifies artifact integrity, and delineates the exact scope of reproduction achieved.

### Key Conclusions:
1. **Local Baseline (Level 1) is Fully Reproduced and Internally Consistent**: All 256 urban validation tiles from `data/small_val_set` have been successfully inferred using the official small ESRGAN checkpoint (`experiments/esrgan_8S2.pth`, 16,710,083 parameters) and evaluated across all primary metrics. Manifests, CSVs, and summary JSONs agree with 100% precision.
2. **Evaluation Pipeline (Level 2) is Validated**: Stride-2 cPSNR, raw PSNR, crop-border-4 PSNR, SSIM, VGG LPIPS distance, and CLIPScore (CLIPA-ViT-bigG-14) implementations are rigorously verified against official code repositories.
3. **Exact Paper Table 2 Reproduction (Level 3) is NOT Achieved**: The published paper's Table 2 evaluated an unreleased **Medium ESRGAN model (~87M parameters)** over a **nationwide 8,192-tile validation benchmark**, reporting LPIPS as **2AFC human-preference agreement accuracy** derived from 11,524 Amazon Mechanical Turk judgments. The local experiment evaluates the **Small ESRGAN model (16.71M parameters)** over **256 urban tiles**, reporting raw **LPIPS perceptual distance**. These two regimes are structurally distinct and cannot be directly equated.

---

## 2. Experimental Setup and Hardware Environment

| Dimension | Audit Specification | Verified Evidence |
| :--- | :--- | :--- |
| **Repository Path** | `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution` | Git working directory intact |
| **Git Commit** | `eb3dcae4cae4492652b04c86cb31d167f8b9d997` | Recorded in `baseline_manifest.json` |
| **Python Environment** | `C:\Users\CSE-SDPL\Desktop\DL_image\.venv` | Python 3.9.6 |
| **Deep Learning Framework** | PyTorch `2.1.0+cu121` | CUDA 12.1 runtime enabled |
| **Accelerator** | NVIDIA RTX 2000 Ada Generation | 16 GB VRAM |
| **Model Checkpoint** | `experiments/esrgan_8S2.pth` | 66,956,559 bytes; SHA-256 verified |
| **Checkpoint SHA-256** | `4099b244799c9ef89679fe26a71cb0a8ce885bb3c3dbff061bc4a7ffaa2d59ac` | Exact match |
| **Architecture** | RRDBNet (`ssr/models/archs/rrdbnet_arch.py`) | 72 input channels, 64 features, 23 RRDB blocks, growth channel 32 |
| **Parameter Count** | **16,710,083** parameters | Reconciled against state dictionary |
| **Validation Dataset** | `data/small_val_set` | 256 tiles (512 files: 256 inputs, 256 targets) |
| **Input Dimensions** | $72 \times 128 \times 128$ | 9 spectral bands $\times$ 8 temporal revisits |
| **Target Dimensions** | $3 \times 512 \times 512$ | 0.5m NAIP high-resolution RGB imagery |
| **Scale Factor** | $4\times$ spatial super-resolution | $128 \times 128 \rightarrow 512 \times 512$ |

---

## 3. Baseline Results and Verification

The empirical baseline metrics recorded in `results/satlas_sr_baseline_freeze_v1/metrics_summary.json` and `results/satlas_sr_revalidation/per_tile_metrics.csv` were verified tile-by-tile across all 256 instances without error:

| Metric | Sample Size ($N$) | Mean | Std Dev | Min | Max | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **cPSNR (crop 4)** | 256 | **20.2144 dB** | 4.1478 dB | 11.2330 dB | 33.1592 dB | Official PROBA-V spatial offset & brightness correction |
| **Raw PSNR (full frame)** | 256 | **18.8122 dB** | 3.8138 dB | 10.9832 dB | 31.0543 dB | Standard unaligned MSE-based PSNR |
| **Crop-Border-4 PSNR** | 256 | **18.8870 dB** | 3.8600 dB | 10.9912 dB | 31.1340 dB | PSNR excluding boundary boundary artifacts |
| **Raw SSIM (full frame)** | 256 | **0.3950** | 0.1495 | 0.0821 | 0.8124 | Structural similarity across full frame |
| **Crop-Border-4 SSIM** | 256 | **0.3912** | 0.1564 | 0.0811 | 0.8105 | Structural similarity excluding 4-pixel border |
| **VGG LPIPS Distance** | 256 | **0.3470** | 0.0809 | 0.1502 | 0.6120 | Raw perceptual feature distance (lower is closer) |
| **CLIPScore (Cosine Sim)** | 256 | **0.9612** | 0.0344 | 0.8241 | 0.9932 | CLIPA-ViT-bigG-14 normalized embedding cosine |

### Artifact Cross-Verification Check:
- Predictions in `results/satlas_sr_full_validation/predictions`: **256 files** (100% complete).
- Ground truth tiles in `data/small_val_set`: **256 files** (100% paired).
- Rows in `per_tile_metrics.csv`: **256 records** (zero missing values, zero NaN entries).
- Reconciled maximum cPSNR-minus-PSNR gain across the 256 tiles is **+12.8842 dB** (mean gain: $+1.3274$ dB). The previously cited $+17.38$ dB figure was an isolated raw RGB tile contact-sheet visualization artifact and has been formally retired.

---

## 4. Metric Definitions and Operational Clarifications

To prevent misleading comparisons or academic errors, the following distinctions must be strictly maintained:

### 1. Corrected PSNR (cPSNR)
- **Implementation**: `ssr/metrics/cpsnr.py`
- **Methodology**: Evaluates super-resolution quality under spatial alignment jitter and radiometric variation, following the PROBA-V challenge protocol. It executes a **stride-2 grid search** over integer spatial offsets within $[-s, +s]$ and optimizes a scalar brightness bias $\beta$ via closed-form mean subtraction.
- **Crop Border**: Standardized at `crop_border = 4` to eliminate boundary convolution padding artifacts.

### 2. Dense vs. Stride-2 cPSNR
- As established in **EXP-001**, dense stride-1 search is an experimental sensitivity analysis. It yields an average gain of $+0.1232$ dB over stride-2 search ($20.3376$ dB vs. $20.2144$ dB).
- **Standard**: The official benchmark standard remains the **stride-2 search** ($20.2144$ dB). Stride-1 results must not be substituted for official baseline comparisons.

### 3. LPIPS Distance vs. Human-Preference Agreement Accuracy
- **Raw LPIPS Distance**: Measures the $L_2$ distance between deep VGG feature embeddings. The local baseline produces a mean distance of **0.3470**.
- **Paper Table 2 LPIPS Metric**: Reports **63.8%**, which represents **2-Alternative Forced Choice (2AFC) Human-Preference Agreement Accuracy** collected from 11,524 pairwise crowd-worker judgments on Amazon Mechanical Turk.
- **Conversion Non-Equivalence**: Raw LPIPS distance cannot be converted into human preference accuracy via a scalar formula (e.g., $1 - \text{LPIPS}$).

### 4. CLIP Cosine Similarity
- Measures the cosine similarity between CLIPA-ViT-bigG-14 deep semantic feature vectors of predicted and target images.
- A score of **0.9612** is a vector cosine in $[-1, 1]$, not a percentage or an accuracy metric.

---

## 5. Synthesis of Experiments EXP-001 through EXP-005

| Experiment | Title | Core Objective | Key Verified Findings |
| :--- | :--- | :--- | :--- |
| **EXP-001** | *cPSNR Grid Search Sensitivity* | Quantify difference between dense stride-1 and official stride-2 search | Dense stride-1 search yields a small, consistent gain of **$+0.1232$ dB** (95% CI: $[0.1018, 0.1477]$ dB). Stride-2 remains the frozen standard. |
| **EXP-002** | *Paper Gap Analysis* | Formally map differences between local baseline and published paper | Identified a 10-dimension structural divergence. Confirmed local model is **Small (16.71M)** on **256 urban tiles**, whereas paper evaluated **Medium (~87M)** on **8,192 multi-biome tiles**. |
| **EXP-003** | *Metric Standardization* | Evaluate validity of transforming raw LPIPS to paper human score | Refuted naive algebraic transformations ($1 - \text{LPIPS} = 0.653$). Standardized the metric protocol to report raw LPIPS distance (**0.3470**) directly. |
| **EXP-004** | *Provenance & Asset Audit* | Audit Git commit history and remote bucket storage for missing assets | Discovered commit `7ccf8d1` migrating assets from Cloudflare R2 to Google Cloud Storage. Verified that the 87M Medium checkpoint was never released to the public. |
| **EXP-005** | *Supplementary PDF Verification* | Inspect authoritative supplementary PDF for evaluation procedures | Retrieved `Zooming_Out_On_Zooming_In_Supplementary.pdf` (1.74 MB, SHA-256 `ebdf117d...`). Verified that Section A.1 documents an empirical AMT crowdsourcing study (11,524 trials), confirming that **no scalar mathematical formula exists** to map individual tile LPIPS distances to human preference scores. |

---

## 6. Reproduction Scope and Explicit Limitations

The reproduction boundaries are categorized into three distinct operational tiers:

```mermaid
flowchart TD
    subgraph L1 [Level 1: Completed Local Baseline]
        A[Official 16.71M Small ESRGAN Checkpoint] --> B[256 Urban Validation Tiles]
        B --> C[Full Metric Evaluation: cPSNR 20.21 dB, LPIPS 0.347]
    end

    subgraph L2 [Level 2: Evaluation Pipeline Reproduction]
        D[PROBA-V Stride-2 cPSNR Algorithm] --> E[Standard PSNR/SSIM & VGG LPIPS]
        E --> F[CLIPA-ViT Cosine Evaluation]
    end

    subgraph L3 [Level 3: Exact Paper Table 2 Reproduction]
        G[Unreleased 87M Medium ESRGAN Checkpoint] -.-> H[8,192 Multi-Biome Validation Tiles]
        H -.-> I[11,524 Amazon Mechanical Turk Human Judgments]
    end

    L1 -->|Fully Verified & Frozen| PASS[PASS]
    L2 -->|Rigorously Validated| PASS
    L3 -->|Blocked by Upstream Data Availability| BLOCKED[NOT ACHIEVED]
```

### Detailed Limitation Matrix:

1. **Model Capacity Mismatch**:
   - *Local Baseline*: Small ESRGAN generator with 16,710,083 parameters.
   - *Paper Table 2*: Medium ESRGAN with ~87M parameters.
   - *Impact*: Local baseline PSNR/cPSNR cannot be directly benchmarked against Table 2 numbers.

2. **Validation Dataset Distribution Mismatch**:
   - *Local Baseline*: 256 tiles exclusively situated in urban regions.
   - *Paper Table 2*: 8,192 tiles stratified across multiple geographic biomes (urban, rural, forest, arid, coastal).
   - *Impact*: Urban imagery typically presents higher high-frequency edge entropy, skewing average PSNR relative to rural/coastal scenes.

3. **Human Preference Data Absence**:
   - *Local Baseline*: Computes mathematical distance across VGG feature spaces.
   - *Paper Table 2*: Evaluates percentage of trials where human evaluators preferred the super-resolved output over bicubic/baseline alternatives.
   - *Impact*: The paper's 63.8% score cannot be replicated without access to the original AMT judgment pairs or executing a new human study.

---

## 7. Conclusions and Recommended Next Steps

### Audit Conclusion
The local Satlas-SR reproduction project has achieved **complete reproducibility and auditability at Level 1 (Local Baseline)** and **Level 2 (Evaluation Pipeline)**. Every claim in the repository is backed by immutable cryptographic hashes, verified manifests, and deterministic execution logs. The gap preventing Level 3 (Paper Table 2) reproduction is conclusively documented as upstream data and model unavailability, rather than experimental error.

### Recommended Next Steps:

1. **Maintain Freeze on Baseline Artifacts (Immediate)**:
   - Preserve `results/satlas_sr_baseline_freeze_v1/` and `results/satlas_sr_experiment_006_final_audit/` as immutable historical baselines.
   - Do not rerun inference or overwrite metric CSV files.

2. **Optional Future Research Directions (Non-Destructive)**:
   - **Direction A (Full Validation Dataset Scaling)**: If storage and network bandwidth permit, download the full 8,192-tile validation set from the Google Cloud Storage bucket identified in EXP-004 to quantify domain transfer across non-urban biomes.
   - **Direction B (Local Model Training/Fine-Tuning)**: If compute resources (multi-GPU) are allocated, train a medium-scale ESRGAN model using `ssr/train.py` to bridge the parameter gap identified in EXP-002.
   - **Direction C (Author Inquiry)**: Optionally contact the authors to request the historical 87M checkpoint weights and the 11,524 Mechanical Turk comparison labels.
