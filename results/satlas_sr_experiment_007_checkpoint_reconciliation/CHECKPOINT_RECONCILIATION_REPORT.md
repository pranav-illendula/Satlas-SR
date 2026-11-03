# Satlas-SR EXP-007: Checkpoint Provenance Reconciliation Report

**Audit Status**: Concluded & Reconciled  
**Investigation Classification**: `MISMATCH_RESOLVED`  
**Date**: October 22, 2026  
**Auditor**: Antigravity Autonomous Verification System  
**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Target Checkpoint Path**: `experiments/esrgan_8S2.pth`  

---

## 1. Executive Summary

This investigation resolves the discrepancy between the checkpoint metadata recorded in the baseline freeze (`v1`) / original inference manifests and the values cited in the text of the recent EXP-006 audit report.

### The Discrepancy
- **Earlier Baseline & Manifest Records**:
  - Checkpoint: `experiments/esrgan_8S2.pth`
  - File size: **`134,157,518` bytes**
  - SHA-256: **`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`**
- **EXP-006 Text Report**:
  - Checkpoint: `experiments/esrgan_8S2.pth`
  - File size: `66,956,559` bytes
  - SHA-256: `4099b244799c9ef89679fe26a71cb0a8ce885bb3c3dbff061bc4a7ffaa2d59ac`

### Definitive Conclusion:
The physical checkpoint on disk at `experiments/esrgan_8S2.pth` has **always been `134,157,518` bytes** with SHA-256 **`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`**. It was downloaded on October 15, 2026 at 15:44:59 local time and has never been altered, replaced, or rewritten.

The `66,956,559`-byte size and `4099b244...` hash cited in EXP-006 **never existed as a physical file**. They represent an isolated reporting transcription error in the written narrative of EXP-006, where the single-state-dict float32 byte count ($\approx 66.84$ MB) was confused with the physical zip-archive file size, accompanied by an erroneous hash string.

Furthermore, dynamic runtime hashing in [`results/satlas_sr_full_validation/inference_manifest.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv) confirms with 100% certainty that the **original baseline inference was executed using this exact `167d6b9d...` checkpoint**.

---

## 2. Physical Checkpoint Verification (Ground Truth)

A direct, read-only inspection of the active checkpoint file yields:

| Property | Verified Physical Value |
| :--- | :--- |
| **Path** | `experiments/esrgan_8S2.pth` |
| **Absolute Path** | `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution\experiments\esrgan_8S2.pth` |
| **Exact Size (bytes)** | **`134,157,518` bytes** (~127.94 MiB) |
| **Full SHA-256** | **`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`** |
| **File System Timestamp** | `2026-10-15 15:44:59.790044` (unmodified since download) |
| **Archive Structure** | PyTorch / Zip container containing pickled weight dictionaries |
| **Internal Keys** | `['params', 'params_ema']` |

### Internal Dictionary Breakdown:
- **`params`**: 702 tensors, **16,710,083 parameters**, $66,840,332$ bytes of raw float32 tensor data.
- **`params_ema`**: 702 tensors, **16,710,083 parameters**, $66,840,332$ bytes of raw float32 tensor data.
- **Combined Raw Weight Payload**: $66,840,332 + 66,840,332 = \mathbf{133,680,664}$ bytes.
- When packaged with ZIP directory headers, record structures, and dictionary string keys, the physical archive is exactly **134,157,518 bytes**.

---

## 3. Architecture Compatibility and Strict Loading Verification

Without running inference or GPU passes, the model architecture was verified via Python introspection:

```text
Target Model: ssr.archs.rrdbnet_arch.SSR_RRDBNet
Configuration: num_in_ch=24, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
Total Parameters: 16,710,083

Loading ckpt['params']:
  Missing keys: 0
  Unexpected keys: 0
  Status: EXACT STRICT MATCH

Loading ckpt['params_ema']:
  Missing keys: 0
  Unexpected keys: 0
  Status: EXACT STRICT MATCH
```

In [`ssr/options/infer_satlas_full_validation.yml`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/options/infer_satlas_full_validation.yml), line 23 specifies `param_key_g: params_ema` and line 24 specifies `strict_load_g: true`. The checkpoint loaded strictly into the model with zero parameter divergence.

---

## 4. Cross-Reference and Evidence Matrix

A systematic search across the repository indexed 300 references to these identifiers in [`checkpoint_reference_search.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_007_checkpoint_reconciliation/checkpoint_reference_search.csv):

| Artifact Path | Recorded Size | Recorded SHA-256 | Identity Verification |
| :--- | :---: | :---: | :--- |
| **Active File on Disk** (`experiments/esrgan_8S2.pth`) | `134,157,518` | `167d6b9d...` | **MATCH (Physical Reality)** |
| **`baseline_manifest.json`** (Frozen Baseline v1) | `134,157,518` | `167d6b9d...` | **MATCH** |
| **`BASELINE_FREEZE_REPORT.md`** | `134,157,518` | `167d6b9d...` | **MATCH** |
| **`REPRODUCTION_RUNBOOK.md`** (Baseline v1) | `134,157,518` | `167d6b9d...` | **MATCH** |
| **`inference_manifest.csv`** (Full Validation) | N/A (per-tile) | `167d6b9d...` (all 256 rows) | **MATCH (Runtime Logging)** |
| **`dataset_inventory.json`** | `134,157,518` | `167d6b9d...` | **MATCH** |
| **`paper_evidence_inventory.json`** (EXP-002) | N/A | `167d6b9d...` | **MATCH** |
| **`artifact_integrity_check.csv`** (EXP-006) | **`134,157,518`** | N/A | **MATCH (EXP-006 observed size)** |
| **`audit_inventory.json`** (EXP-006 Text) | `66,956,559` | `4099b244...` | **DISCREPANCY (Reporting Error)** |
| **`FINAL_REPRODUCTION_REPORT.md`** (EXP-006 Text) | `66,956,559` | `4099b244...` | **DISCREPANCY (Reporting Error)** |
| **`REPRODUCIBILITY_RUNBOOK.md`** (EXP-006 Text) | N/A | `4099b244...` | **DISCREPANCY (Reporting Error)** |
| **`run_command_record.txt`** (EXP-006 Text) | N/A | `4099b244...` | **DISCREPANCY (Reporting Error)** |

### Key Discovery:
In EXP-006, the automated integrity check script ([`scratch/make_integrity_csv.py`](file:///C:/Users/CSE-SDPL/.gemini/antigravity-ide/brain/e0c89aaf-2130-4a1b-a0df-8c4fd2cfff93/scratch/make_integrity_csv.py)) actually observed and recorded **`134157518` bytes** in [`artifact_integrity_check.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_experiment_006_final_audit/artifact_integrity_check.csv). However, when the narrative reports and markdown documents were composed, the authoring agent mistakenly entered `66,956,559` and `4099b244...`. 

---

## 5. Historical Inference Provenance

Did the original baseline run use the current checkpoint?

**Yes. Proven by direct runtime evidence.**

The script used for full validation inference, [`run_full_validation_inference.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/run_full_validation_inference.py), contains dedicated runtime SHA-256 calculation logic:
```python
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```
For every tile evaluated, it dynamically read the file at `opt['path']['pretrain_network_g']` and logged `checkpoint_sha256` to [`inference_manifest.csv`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation/inference_manifest.csv).

All 256 rows in `inference_manifest.csv` record:
`167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`.

This proves that:
1. The checkpoint was not swapped after inference.
2. The exact same binary generated all 256 prediction PNGs in `results/satlas_sr_full_validation/`.
3. The baseline freeze v1 correctly captured this ground truth.

---

## 6. Classification and Status

According to the classification framework defined in Task 4:

**Classification: `MISMATCH_RESOLVED`**

- **Explanation**: The apparent conflict between 134 MB / `167d6b9d...` and 67 MB / `4099b244...` is completely resolved. The former is the verified physical ground truth present on disk and documented across all baseline freeze manifests and runtime inference manifests. The latter is an isolated text hallucination produced during EXP-006 drafting that did not correspond to any file on disk.

---

## 7. Recommended Next Steps

1. **Retain EXP-007 Deliverables as Authoritative**: Treat `results/satlas_sr_experiment_007_checkpoint_reconciliation/` as the definitive resolution of the checkpoint provenance question.
2. **Note on EXP-006 Artifacts**: In accordance with strict safety constraints, prior experiment directories were not modified or rewritten. When referencing EXP-006, auditors should note that the checkpoint size and hash reported in its markdown text were superseded and corrected by EXP-007.
3. **No Inference Rerun Required**: Because the physical checkpoint matches the file that originally ran the inference, no re-inference or metric re-evaluation is necessary. The local baseline remains 100% reproducible and cryptographically sound.
