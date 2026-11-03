# Satlas-SR EXP-011: Image Resolution and Output Shape Reconciliation

**Document Status**: Official Reconciliation Report  
**Date**: October 22, 2026  
**Auditor**: Antigravity Autonomous Verification System  
**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  

---

## 1. Executive Summary: The 128×128 vs. 512×512 Discrepancy

In several earlier user guides and high-level report narratives, the Satlas-SR super-resolution task was described as taking $128 \times 128$ inputs and generating $512 \times 512$ high-resolution predictions. However, direct inspection of the physical validation artifacts reveals that the saved super-resolution predictions in [`results/satlas_sr_full_validation/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/results/satlas_sr_full_validation) are **$128 \times 128$ pixels**, and the ground truth NAIP aerial images in [`data/small_val_set/naip/`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/data/small_val_set/naip) are also **$128 \times 128$ pixels**.

This document reconciles this apparent discrepancy by tracing the mathematical tensor transformations through the model architecture, the dataset preprocessing code, and the physical files stored on disk.

---

## 2. Evidence from Physical Files on Disk

Direct verification of the image files in `small_val_set` and `satlas_sr_full_validation` using Python's Pillow library establishes:

| File Target | Path | Physical Dimensions $(W \times H)$ | Channels / Mode | Explanation |
| :--- | :--- | :---: | :---: | :--- |
| **Sentinel-2 Input** | `data/small_val_set/sentinel2/20771_46525/tci.png` | **$32 \times 1024$** | RGB (3-channel) | 32 frames of $32 \times 32$ pixels stacked vertically ($32 \times 32 = 1024$) |
| **NAIP Target Reference** | `data/small_val_set/naip/.../20771_46525/rgb.png` | **$128 \times 128$** | RGB (3-channel) | High-resolution aerial reference image |
| **Saved Baseline SR Output** | `results/satlas_sr_full_validation/20771_46525/sr.png` | **$128 \times 128$** | RGB (3-channel) | Model super-resolution prediction |
| **Saved Baseline LR Reference**| `results/satlas_sr_full_validation/20771_46525/lr.png` | **$32 \times 32$** | RGB (3-channel) | First low-resolution frame saved for visual comparison |

---

## 3. Mathematical Tensor Trace Through Architecture

The tensor transformations through [`ssr/utils/infer_utils.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/utils/infer_utils.py) and [`ssr/archs/rrdbnet_arch.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/archs/rrdbnet_arch.py) are completely deterministic:

```
+----------------------------------------------------------------------------------------------------+
|                                    TENSOR SHAPE FLOW (ESRGAN 8S2)                                  |
+----------------------------------------------------------------------------------------------------+
  1. Input PNG File:
     s2_data shape: [1024, 32, 3]  (32 vertical slices of 32x32 RGB)

  2. Chunking & Reshaping (infer_utils.py line 8):
     s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))  --> Shape: [32, 32, 32, 3]

  3. Frame Sampling (infer_utils.py line 25):
     Select 8 frames out of 32  --> Shape: [8, 32, 32, 3]

  4. Channel Concatenation & Batching (infer_utils.py lines 34-36):
     Permute to [3, 32, 32] and concatenate 8 frames along channels:
     input_tensor shape: [1, 24, 32, 32]  (where 24 = 8 frames * 3 RGB channels)

  5. Generator Forward Pass (SSR_RRDBNet, scale=4):
     - conv_first: [1, 24, 32, 32] --> [1, 64, 32, 32]
     - 23 RRDB blocks: [1, 64, 32, 32] --> [1, 64, 32, 32]
     - conv_body: [1, 64, 32, 32] --> [1, 64, 32, 32]
     - upsample 1 (2x Nearest + Conv): [1, 64, 32, 32] --> [1, 64, 64, 64]
     - upsample 2 (2x Nearest + Conv): [1, 64, 64, 64] --> [1, 64, 128, 128]
     - conv_hr & conv_last: [1, 64, 128, 128] --> [1, 3, 128, 128]

  6. Output Prediction:
     prediction shape: [1, 3, 128, 128]  --> Squeezed & transposed to [128, 128, 3]
+----------------------------------------------------------------------------------------------------+
```

### Architectural Verification:
- In `ssr/options/infer_satlas_full_validation.yml` line 4: `scale: 4`.
- Input spatial dimensions: $H_{\text{in}} = 32, W_{\text{in}} = 32$.
- Expected spatial dimensions:
  $$H_{\text{out}} = H_{\text{in}} \times \text{scale} = 32 \times 4 = \mathbf{128}$$
  $$W_{\text{out}} = W_{\text{in}} \times \text{scale} = 32 \times 4 = \mathbf{128}$$
- In `run_full_validation_inference.py` lines 204-205:
  ```python
  if prediction.shape != (3, 128, 128):
      raise ValueError(f"Expected model output (3,128,128), got {prediction.shape}")
  ```
- In `run_full_validation_inference.py` line 211:
  ```python
  if not image_is_valid(Path(row["sr_output_path"]), (128, 128)):
      raise ValueError("Written output failed PNG/RGB/128x128 validation")
  ```

The pipeline explicitly enforces and validates that the model output is strictly $(128, 128)$.

---

## 4. Origin of the 512×512 Figure in Literature and Prior Reports

Why did earlier narratives and high-level summaries refer to $512 \times 512$?

1. **Large Grid Stitching (`stitch` function in `infer_utils.py`)**:
   In `ssr/utils/infer_utils.py` lines 42-62, a utility function `stitch()` is defined:
   ```python
   def stitch(chunks_dir, img_size, save_path, scale=4, grid_size=16, sentinel2=False):
   ```
   When super-resolving a large continuous geographical area, the authors partition a large $512 \times 512$ NAIP tile into a $16 \times 16$ grid of small $32 \times 32$ chunks ($16 \times 32 = 512$). Each chunk is super-resolved to $128 \times 128$, and then stitched together into a $2048 \times 2048$ image (or alternatively, a $128 \times 128$ input grid stitched to $512 \times 512$).
2. **Standard Multi-Spectral Imagery in Satellite Literature**:
   In remote sensing super-resolution papers (such as HighRes-net or standard Real-ESRGAN benchmarks), test chips are frequently cut to $128 \times 128$ low-resolution, which at $4\times$ upsampling yields $512 \times 512$ high-resolution outputs.
3. **Drafting Conflation**:
   In earlier high-level narrative summaries (including EXP-006 and EXP-008 text overviews), the phrase "128x128 input to 512x512 target" was repeated as a convention from generic super-resolution literature without checking the specific tile resolution of `small_val_set`.

---

## 5. Ground Truth Alignment Confirmation

Both the predicted super-resolution image (`sr.png`) and the ground truth NAIP reference image (`rgb.png`) have identical spatial dimensions of **$128 \times 128$ pixels**:
- Spatial Footprint: At Sentinel-2 native resolution ($10\text{ m/pixel}$), a $32 \times 32$ tile covers a ground area of:
  $$32 \text{ pixels} \times 10\text{ m/pixel} = 320\text{ m} \times 320\text{ m}$$
- Target Footprint: The ground truth NAIP chip is cropped to $128 \times 128$ pixels, covering the exact same $320\text{ m} \times 320\text{ m}$ ground footprint at an effective grid spacing of:
  $$\frac{320\text{ m}}{128\text{ pixels}} = 2.5\text{ m/pixel}$$
- When `evaluate_satlas_sr_paper_metrics.py` compares `sr.png` against `rgb.png`, both arrays have shape `(128, 128, 3)` with zero padding or resizing needed.

---

## 6. Conclusion

- **Verified Reality**: The input is $32 \times 32$, the scale is $4\times$, the model prediction is $128 \times 128$, and the ground truth target is $128 \times 128$.
- **No Pipeline Modification Needed**: The local code and model architecture are completely self-consistent. The $512 \times 512$ narrative was a descriptive misnomer that did not affect the actual code, image files, or metric calculations.
