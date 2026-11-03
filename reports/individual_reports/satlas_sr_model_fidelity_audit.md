# Satlas-SR Model Architecture & Preprocessing Fidelity Audit Report

- **Date**: 2026-10-22
- **Audit Target**: `experiments/esrgan_8S2.pth`, `run_full_validation_inference.py`, `ssr/infer.py`, `ssr/models/ssr_rrdbnet_arch.py`, and `ssr/utils/infer_utils.py`
- **Official Checkpoint SHA-256**: `167d6b9d5c931be06bde6cfbdd62edf5f1ac8aa9b357c4146c7e948d3f672e07`

---

## 1. Specification & Architecture Verification

| Component | Paper / Official Specification | Local Implementation | Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Generator Architecture** | `SSR_RRDBNet` (Multi-frame ESRGAN) | `SSR_RRDBNet` | `ssr/models/ssr_rrdbnet_arch.py` loaded via `build_network(opt)` | **PASS (Exact)** |
| **Input Channels** | $24$ (8 frames $\times$ 3 RGB channels) | $24$ (`num_in_ch: 24`) | Verified in config & model input tensor `(1, 24, 32, 32)` | **PASS (Exact)** |
| **Output Channels** | $3$ (Single high-res RGB image) | $3$ (`num_out_ch: 3`) | Verified in model output tensor `(1, 3, 128, 128)` | **PASS (Exact)** |
| **Feature Channels** | $64$ | $64$ (`num_feat: 64`) | Internal conv layers have 64 filters | **PASS (Exact)** |
| **RRDB Blocks** | $23$ Residual-in-Residual Dense Blocks | $23$ (`num_block: 23`) | 23 RRDB blocks in body sequential container | **PASS (Exact)** |
| **Growth Channels** | $32$ | $32$ (`num_grow_ch: 32`) | Dense connections grow by 32 per sub-layer | **PASS (Exact)** |
| **Scale Factor** | $4\times$ spatial super-resolution | $4\times$ (`scale: 4`) | S2 $32 \times 32 \to$ NAIP $128 \times 128$ | **PASS (Exact)** |
| **Total Parameters** | $16,707,331$ (~$16.71\text{M}$) | $16,707,331$ | PyTorch `sum(p.numel() for p in model.parameters())` | **PASS (Exact)** |

---

## 2. Checkpoint & Parameter Weights Audit

1. **Checkpoint Contents**:
   - The official checkpoint `experiments/esrgan_8S2.pth` (134 MB) contains two root keys: `['params', 'params_ema']`.
   - Both dictionaries contain exactly 702 state dict entries.
2. **`params` vs. `params_ema`**:
   - `params`: Raw training generator weights from the final iteration.
   - `params_ema`: Exponential Moving Average (EMA) of generator parameters accumulated during training.
   - **Configuration Choice**: The official evaluation configuration specifies `param_key_g: params_ema`. EMA weights are standard in ESRGAN/diffusion literature to suppress high-frequency generative artifacts and stabilize validation performance.
3. **Strict Checkpoint Loading**:
   - Both `ssr/infer.py` and `run_full_validation_inference.py` call:
     ```python
     model.load_state_dict(checkpoint['params_ema'], strict=True)
     ```
   - Zero missing keys and zero unexpected keys were reported. Strict parameter loading succeeded 100%.

---

## 3. Preprocessing & Input Data Pipeline Audit

### 3.1 Eight-Frame Selection & Cloud/Black Filtering
In `ssr/utils/infer_utils.py`:
- Each Sentinel-2 tile consists of 32 temporal acquisitions stacked vertically into a $(1024, 32, 3)$ PNG.
- The pipeline splits the PNG into 32 chunks of $(32, 32, 3)$.
- **Black-pixel filter**: Chunks containing unpopulated/invalid pixels (`[0, 0, 0]`) are separated into `bads`; clean chunks are separated into `goods`.
- **Random sample**: If `len(goods) >= 8`, 8 chunks are sampled uniformly at random from `goods`.
- **Reproducibility & Determinism**:
  - `ssr/infer.py` sets `random.seed(opt.get('manual_seed', 0))` but iterates over unsorted `glob.glob()`. Because filesystem glob ordering differs across operating systems, the RNG sequence could assign different frames to different tiles across runs.
  - `run_full_validation_inference.py` sorts tile IDs deterministically (`sorted(s2_dirs)`), records the exact selected zero-based frame indices in `inference_manifest.csv`, and validates that resuming identical runs reproduces exact frame selections.

### 3.2 Color Channel Order & Normalization
- Sentinel-2 PNGs are loaded in standard RGB order via `skimage.io.imread`.
- Chunks are permuted from `(H, W, C)` to `(C, H, W)`.
- Chunks are concatenated along the channel dimension into a tensor of shape `(1, 24, 32, 32)`.
- Input normalization:
  $$\text{input\_tensor} = \frac{\text{tensor}}{255.0} \in [0.0, 1.0]$$
- No zero-mean standardization is applied to inputs, strictly matching official model training.

---

## 4. Postprocessing & Output Generation Audit

1. **Clamping**:
   - The raw model output is clamped to $[0.0, 1.0]$ via `torch.clamp(output, 0, 1)`.
2. **Quantization & Channel Transposition**:
   - Squeezed to $(3, 128, 128)$ and transposed to $(128, 128, 3)$.
   - Quantized to 8-bit unsigned integer via `(output * 255).astype(np.uint8)`.
3. **Image Saving**:
   - Saved as uncompressed PNG using `skimage.io.imsave(path, output, check_contrast=False)`.
   - Verified that every output image has format `PNG`, mode `RGB`, and dimensions `(128, 128)`.

---

## 5. Comparison: `run_full_validation_inference.py` vs. `ssr/infer.py`

| Dimension | Official `ssr/infer.py` | Reproduction `run_full_validation_inference.py` | Deviation / Impact |
| :--- | :--- | :--- | :--- |
| **Model Weights & Loading** | `build_network` + `load_state_dict(strict=True)` | `build_network` + `load_state_dict(strict=True)` | None (Exact match) |
| **Preprocessing Function** | `format_s2naip_data` | `infer_utils.format_s2naip_data` | None (Directly imports official code) |
| **Forward Pass** | `model(input_tensor)` | `model(input_tensor)` with `torch.inference_mode()` | None (Numerically identical) |
| **Postprocessing** | Clamp $[0, 1] \to \times 255 \to \text{uint8}$ | Clamp $[0, 1] \to \times 255 \to \text{uint8}$ | None (Exact match) |
| **Tile Ordering** | Non-deterministic `glob.glob` | Deterministically sorted by tile ID | **Improvement**: Guarantees deterministic RNG stream |
| **Traceability** | Saves to anonymous indices `0, 1, 2...` | Preserves geographic tile ID folder names | **Improvement**: Enables 1-to-1 pairing with NAIP reference |
| **Execution Manifest** | None | Atomic CSV tracking SHA-256 hashes & frame indices | **Improvement**: Complete scientific audit trail |

**Audit Conclusion**: The inference pipeline strictly implements the official architecture, weights, preprocessing, and postprocessing with zero algorithmic shortcuts or quality degradation.
