# Research Next Step Analysis & Recommendation (v1.0)

- **Date**: 2026-10-22
- **Author**: Senior Research Engineer (Antigravity AI)
- **Context**: Satlas-SR Baseline Freeze Review (`results/satlas_sr_baseline_freeze_v1/`)
- **Output Document**: `results/satlas_sr_baseline_freeze_review_v1/RESEARCH_NEXT_STEP.md`

---

## 1. Evaluation of Candidate Research Paths

Following the successful freeze of the 256-tile urban baseline for the released small ESRGAN model (`experiments/esrgan_8S2.pth`), two distinct research avenues present themselves:

```
                                      [Frozen Baseline (v1.0)]
                                      256 Tiles | 16.71M ESRGAN
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
            [Option A: Paper-Alignment]                     [Option B: Controlled Improvement]
         Nationwide Multi-Biome Benchmark                 Controlled Experiment Against Baseline
        (Requires large data / 87M weights)              (Zero external downloads, immediate ROI)
```

---

### Option A: Paper-Alignment Phase (Nationwide Benchmark Reconstruction)

#### Objective
Investigate the full evaluation protocol from *Satlas-SR: A Dataset and Benchmark for Satellite Super-Resolution* ([arXiv:2311.18082](https://arxiv.org/abs/2311.18082)), characterize the exact multi-biome test splits, and establish the requirements for evaluating the 87M medium ESRGAN model and SR3 diffusion baseline.

#### Required Work
1. Inspect paper data split manifests (e.g., test tile lists across US regions).
2. Document the exact data ingestion pipeline for full-resolution nationwide Sentinel-2 / NAIP pairs.
3. Analyze checkpoint availability for the 87M medium ESRGAN and 97M SR3 models.

#### Constraints & Blockers
- **Data Volume**: The full nationwide benchmark encompasses thousands of tiles and dozens of gigabytes of raw data. Downloading this benchmark without prior user authorization violates our safety directives.
- **Compute Requirements**: Re-evaluating large models (87M ESRGAN, 97M SR3 iterative diffusion) across thousands of tiles requires extensive compute time exceeding local interactive sessions.
- **Immediate Outcome**: Option A can only produce an architectural/data audit document; it cannot execute a live empirical comparison without acquiring large external assets.

---

### Option B: Controlled Improvement Phase (Targeted Experiment on Frozen Baseline)

#### Objective
Design and execute a tightly controlled, reproducible experiment directly against the frozen 256-tile baseline, evaluating a specific algorithmic hypothesis on the existing predictions without touching baseline files or downloading external assets.

#### Candidate Controlled Experiment: EXP-001 (Dense Stride-1 cPSNR Search)
- **Scientific Rationale**:
  - In our independent review, we identified that the official PROBA-V cPSNR implementation evaluates only even-pixel shifts:
    $$\Delta \in \{-8, -6, -4, -2, 0, +2, +4, +6, +8\}^2 \quad (81\text{ offsets, stride = 2})$$
  - In earlier reports, an unmeasured claim was made that this stride underestimates alignment by "0.2–0.5 dB" on odd-shifted tiles.
  - **The Hypothesis**: Replacing the stride-2 search with a dense stride-1 grid (evaluating all 289 integer offsets in $[-8, +8] \times [-8, +8]$) will systematically recover additional fidelity on tiles with odd-integer misregistration, establishing the true quantitative impact of the stride limitation across all 256 tiles.
- **Experimental Protocol**:
  1. Input: Exact 256 super-resolved predictions in `results/satlas_sr_full_validation/`.
  2. Ground Truth: Exact 256 NAIP reference images in `data/small_val_set/naip/`.
  3. Metric Comparison: Compare baseline cPSNR (stride 2, 81 offsets) vs. dense cPSNR (stride 1, 289 offsets).
  4. Isolation: Zero retraining, zero re-inference, zero external downloads.
  5. Deliverables: Formally recorded in a new entry under `results/experiments/exp_001_dense_cpsnr/` using `EXPERIMENT_TEMPLATE.md`.

---

## 2. Comparative Evaluation Matrix

| Evaluation Dimension | Option A: Paper-Alignment Phase | Option B: Controlled Improvement Phase |
| :--- | :--- | :--- |
| **Safety & Non-Destructiveness** | High (Read-only inspection) | **Very High** (Isolated to new `results/experiments/` dir) |
| **External Dependencies** | High (Requires large dataset/model access) | **Zero** (100% self-contained in local repository) |
| **Scientific Value** | Documents gap to paper | **Directly resolves an empirical unmeasured caveat** |
| **Compute Overhead** | High if executed (hours of GPU time) | **Minimal** (CPU/GPU metric search in ~60 seconds) |
| **Reproducibility** | Dependent on external servers | **100% deterministic and verifiable locally** |

---

## 3. Definitive Recommendation

**Option B (Controlled Improvement Phase) is strongly recommended.**

### Why Option B is the Superior Choice
1. **Direct Scientific Continuation**: Our independent review established that the "0.2–0.5 dB odd-pixel loss" was an unsupported theoretical conjecture. Option B provides the exact empirical measurements needed to validate or refute this claim with high statistical rigor.
2. **Immediate Actionability**: It leverages the newly created `EXPERIMENT_TEMPLATE.md` and tests the end-to-end experimental harness without risking baseline mutation or requiring large multi-gigabyte downloads.
3. **High Signal-to-Cost Ratio**: The experiment can run in under two minutes, evaluate all 256 tiles, and yield a definitive comparison table showing the exact distribution of cPSNR gains across even vs. odd shifts.

*Note: Option B is fully designed and documented above, but has NOT been executed, preserving the read-only directive of this task.*
