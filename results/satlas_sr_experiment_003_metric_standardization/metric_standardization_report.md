# Satlas-SR EXP-003: Paper Metric Standardization Report

- **Date**: 2026-10-22
- **Experiment**: `EXP-003`
- **Evaluator Script**: `scripts/experiment_003_lpips_agreement.py`
- **Target Paper**: *Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing* ([arXiv:2311.18082v1](https://arxiv.org/abs/2311.18082))
- **Execution Time**: 0.02 seconds

## 1. Executive Summary & Audit Outcome

The objective of EXP-003 is to standardize local evaluation metrics against the published Satlas-SR paper, specifically investigating the **LPIPS-derived human-preference agreement transformation** reported in Table 2.

### Definitive Audit Finding
> **STATUS: IMPLEMENTATION BLOCKED BY MISSING EVIDENCE**
>
> In strict adherence to scientific safety controls, **no mathematical transformation was guessed or fabricated**. Inspection of the published arXiv text (arXiv:2311.18082v1) confirms that the paper delegates the LPIPS conversion procedure to **Supplementary Section A.1**. The authoritative URL hosting the supplementary PDF (`https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/Zooming_Out_On_Zooming_In_Supplementary.pdf`) is hosted on a private Cloudflare R2 bucket and returns **HTTP Error 401: Unauthorized**. Because the exact closed-form equation and calibration parameters are unavailable locally and inaccessible externally without credentials, applying any substitute transformation would produce fabricated results.

## 2. Paper Evidence & Caption Analysis

### 2.1 The Table 2 Caption
In the published paper, Table 2 presents results for four methods across four datasets. Its official caption reads verbatim:

> *"Table 2: Results for four methods on four datasets. **The reported LPIPS scores are converted to an accuracy, so with each of these metrics, higher is better.** ESRGAN achieves the best CLIPScore for all datasets, and that is the metric we are prioritizing for this study."*

### 2.2 Table 2 Published Values (S2-NAIP Dataset)

| Method | cPSNR (dB) $\uparrow$ | LPIPS (Converted Accuracy) $\uparrow$ | CLIPScore $\uparrow$ |
| :--- | :---: | :---: | :---: |
| **SRCNN** | 17.9039 | 0.3517 | 0.7000 |
| **HighResNet** | 20.9171 | 0.1990 | 0.6705 |
| **ESRGAN (Medium 87M)** | **22.6506** | **0.8406** | **0.8745** |
| **SR3 (Diffusion 97M)** | 19.4706 | 0.6292 | 0.8223 |

### 2.3 Refutation of Common Heuristics

1. **Why `Agreement = 1 - LPIPS` is Mathematically Invalid**:
   - If the conversion were $1 - \text{LPIPS}$, then HighResNet's raw LPIPS would be $1 - 0.1990 = 0.8010$, and ESRGAN's raw LPIPS would be $1 - 0.8406 = 0.1594$.
   - However, on PROBA-V in Table 2, the published LPIPS values are: SRCNN = `0.2112`, HighResNet = `0.2137`, ESRGAN = `0.2154`, SR3 = `0.2115`. If $1 - \text{LPIPS}$ were used, all four models on PROBA-V would have raw LPIPS distances around $0.785$, which does not align with PROBA-V literature.
   - Furthermore, in Section 3.1, the paper describes a **pairwise 2AFC human judgment study** (11,524 pairwise comparisons on Amazon Mechanical Turk). On this task, **50% accuracy represents random guessing**. Table 2's LPIPS accuracy scores are derived from agreement with human preferences on this pairwise choice dataset, not a simple scalar inversion of single-image perceptual distance.
2. **Why Arbitrary Curve Fitting is Unscientific**:
   - Without the pairwise image pool, worker annotation logs, and exact logistic regression parameters from Supplementary Section A.1, fitting an arbitrary sigmoid or linear curve to map $0.347 \to 0.8406$ would be entirely circular and falsify the audit trail.

## 3. Local Dataset LPIPS Validation (256 Validation Tiles)

The input validation pipeline verified all 256 existing tile records in `results/satlas_sr_revalidation/per_tile_metrics.csv`:

- **Total Records**: 256 unique tiles.
- **Numeric Validity**: 100% of values are valid, non-negative, finite floating-point numbers.
- **Missing / NaN / Inf**: 0 records.

### Raw Perceptual Distance Distribution (`lpips_vgg`)

- **Mean**: **0.34697**
- **Standard Deviation**: 0.08088
- **Median**: **0.33959**
- **Interquartile Range [IQR]**: [0.29843, 0.38182]
- **Range**: [0.14425, 0.71959]
- **Metric Direction**: Lower is better (raw VGG perceptual distance).

## 4. Comparability Boundaries & Methodological Separation

To prevent misinterpretation in future research, the following distinctions are formally established:

1. **Implementation Equivalence vs. Benchmark Comparability**:
   - The local codebase implements official BasicSR/PyTorch VGG16 LPIPS (`ssr/metrics/lpips.py`), which produces standard perceptual distance.
   - The paper's Table 2 presents an aggregated human-preference agreement accuracy score from a pairwise preference study.
2. **Model Capacity Disparity**:
   - Even if the exact conversion formula were obtained, the local baseline evaluates the **16.71M Small ESRGAN**, whereas Table 2 evaluates the **87M Medium ESRGAN**.
3. **Dataset Scope Disparity**:
   - The local evaluation is on the 256-tile urban subset, whereas Table 2 evaluated the full nationwide multi-biome test set.
4. **cPSNR Preservation**:
   - In accordance with experimental safety controls, the official stride-2 cPSNR implementation remains the authoritative baseline metric. The EXP-001 dense stride-1 search ($+0.1232\text{ dB}$ gain) remains an independent diagnostic on translational quantization.

## 5. Required Next Action to Resolve Missing Evidence

To unblock the LPIPS human-preference agreement conversion without guessing:
1. Submit a formal provenance request to the paper maintainers (`piperw@allenai.org`) or repository issue tracker requesting public access to `Zooming_Out_On_Zooming_In_Supplementary.pdf`.
2. Request the specific mathematical equation, calibration coefficients, and sample calculation for converting raw model LPIPS distance into Table 2's agreement accuracy.
3. Retain raw LPIPS distance as the primary verified perceptual metric in all baseline reporting until authoritative provenance is supplied.
