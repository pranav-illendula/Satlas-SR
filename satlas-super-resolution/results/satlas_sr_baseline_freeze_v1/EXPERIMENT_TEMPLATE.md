# Experiment Entry: [EXP-XXX] [Short Title]

- **Date**: YYYY-MM-DD
- **Author**: [Researcher Name / Agent]
- **Target Baseline**: `results/satlas_sr_baseline_freeze_v1/baseline_manifest.json`
- **Baseline Git Commit**: `3e67973f6746bedbd91cc032b4888574980e1daf`
- **Baseline Checkpoint SHA-256**: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`

---

## 1. Hypothesis & Scientific Motivation
*What question does this experiment address? What change is being tested, and what outcome is expected relative to the frozen baseline?*

- **Hypothesis**: 
- **Expected Impact**: 

---

## 2. Experimental Configuration & Modifications
*List all modifications to code, configuration YAMLs, model architectures, loss functions, or data pipelines.*

- **Modified Files**:
  - `path/to/file1.py`: [Description of change]
  - `path/to/config.yml`: [Description of change]
- **Configuration Diff / Key Settings**:
  ```yaml
  # Relevant configuration excerpt
  manual_seed: 0
  n_lr_images: 8
  ```
- **Checkpoint Used**: [Official released `experiments/esrgan_8S2.pth` OR New trained checkpoint path + SHA-256]
- **Model Parameter Count**: [Count parameters directly using PyTorch]

---

## 3. Dataset & Sampling Protocol
- **Dataset Root**: `data/small_val_set`
- **Evaluated Tiles**: [Full 256 tiles OR specific subset of tile IDs]
- **Random Seed**: `0` (or specified seed)
- **Frame Selection Protocol**: [Standard black-pixel filtering + uniform random sample OR custom temporal policy]

---

## 4. Execution Commands
*Document the exact, self-contained commands executed for this experiment.*

```powershell
# 1. Inference Command
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" run_full_validation_inference.py `
    -opt ssr/options/[config_file].yml `
    --output-dir results/experiments/[EXP-XXX]/predictions

# 2. Metric Evaluation Command
& "C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe" evaluate_satlas_sr_paper_metrics.py `
    --predictions-dir results/experiments/[EXP-XXX]/predictions `
    --naip-dir ../data/small_val_set/naip `
    --output-dir results/experiments/[EXP-XXX]/metrics `
    --crop-border 4 `
    --clip-model clipa-ViT-bigG-14
```

---

## 5. Quantitative Results vs. Frozen Baseline

Compare the observed metrics directly against the frozen baseline aggregates on `small_val_set`:

| Metric | Frozen Baseline (v1.0) | Experiment [EXP-XXX] | Absolute Delta ($\Delta$) | Statistically Significant? |
| :--- | :--- | :--- | :--- | :--- |
| **cPSNR (crop 4, dB)** | $20.2144 \pm 4.1478$ | | | [ ] Yes  [ ] No |
| **Standard PSNR (crop 4, dB)** | $18.8870 \pm 3.8600$ | | | [ ] Yes  [ ] No |
| **cPSNR Gain (dB)** | $+1.3274 \pm 2.0991$ | | | [ ] Yes  [ ] No |
| **SSIM (crop 4)** | $0.3912 \pm 0.1564$ | | | [ ] Yes  [ ] No |
| **LPIPS (VGG16)** | $0.3470 \pm 0.0809$ | | | [ ] Yes  [ ] No |
| **CLIPScore (ViT-bigG)** | $0.9612 \pm 0.0344$ | | | [ ] Yes  [ ] No |

*Note: CLIPScore is a cosine similarity metric in $[0.73, 1.00]$, not an accuracy percentage.*

---

## 6. Execution Runtime & Hardware
- **GPU**: NVIDIA RTX 2000 Ada Generation (16 GB VRAM)
- **Execution Time**: [Total seconds / minutes]
- **Average Latency per Tile**: [Seconds per 128x128 tile]
- **Peak GPU Memory Allocated**: [MiB]

---

## 7. Failures, Deviations & Unresolved Issues
- **Failed Tiles**: [List any tile IDs that failed or produced NaNs/Inf]
- **Deviations from Protocol**: [Document any differences from the baseline protocol]
- **Artifacts & Observations**: [Visual inspection notes, checkerboard artifacts, hallucinations, etc.]

---

## 8. Conclusion & Reproducibility Assessment
- **Hypothesis Validated?**: [ ] Yes  [ ] Partial  [ ] No
- **Reproducible?**: [ ] Yes  [ ] Requires verification
- **Recommendation for Next Step**: 
