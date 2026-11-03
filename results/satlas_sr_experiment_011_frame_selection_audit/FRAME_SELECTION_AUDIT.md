# EXP-011: Sentinel-2 Frame-Selection Sensitivity Audit

**Repository**: `C:\Users\CSE-SDPL\Desktop\DL_image\satlas-super-resolution`  
**Execution Environment**: `C:\Users\CSE-SDPL\Desktop\DL_image\.venv\Scripts\python.exe` (PyTorch 2.1.0+cu121)  
**Date**: 2026-10-22  
**Audit Scope**: Strict read-only audit of the Sentinel-2 frame selection and preprocessing implementation, local metadata inventory, determinism analysis, and selection mechanics.

---

## 1. Executive Summary

This audit establishes the exact programmatic pipeline through which multi-temporal Sentinel-2 observations are ingested, filtered, sampled, preprocessed, and fed to the Satlas-SR super-resolution network (`SSR_RRDBNet`). 

Key verified facts:
1. **Physical TCI Image Structure**: Each Sentinel-2 `tci.png` tile in `data/small_val_set/sentinel2/<tile_id>/` has dimensions $32 \times 1024$ pixels in RGB mode, consisting of 32 vertical sub-frames of size $32 \times 32 \times 3$.
2. **Observation Validity**: Across all 256 validation tiles, 100% of the 32 frames have zero black pixels `[0, 0, 0]`. Therefore, `len(goods) == 32` and `len(bads) == 0` universally across the validation dataset.
3. **Selection Logic**: The pipeline calls `random.sample(goods, 8)`. Because all 32 frames are good, this selects an unordered subset of 8 distinct indices from $[0..31]$.
4. **Permutation Sensitivity**: The ESRGAN generator (`SSR_RRDBNet`) ingests a tensor of shape $[1, 24, 32, 32]$ where the first convolution layer `conv_first` has kernel size $(3 \times 3)$ and in-channels 24. Standard 2D convolution is **not permutation-invariant** across input channels. Thus, the order of frames in the 8-tuple directly modulates feature extraction alongside which frames were chosen.
5. **Determinism**: In `run_full_validation_inference.py`, `random.seed(0)` is invoked once at line 165 prior to iterating through 256 tiles in sorted alphanumeric order. The selection is deterministic for full sequential runs, but tile-order dependent.
6. **Local Metadata Absence**: Zero timestamps, cloud scores, solar angles, or acquisition metadata exist on disk in `data/small_val_set`. All files are raw PNG images.

---

## 2. Pipeline Code Walkthrough & Line-by-Line Inspection

The active frame-selection and preprocessing pipeline spans four primary modules:
- [`ssr/utils/infer_utils.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/utils/infer_utils.py)
- [`run_full_validation_inference.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/run_full_validation_inference.py)
- [`ssr/data/s2-naip_dataset.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/data/s2-naip_dataset.py)
- [`ssr/utils/data_utils.py`](file:///C:/Users/CSE-SDPL/Desktop/DL_image/satlas-super-resolution/ssr/utils/data_utils.py)

### 2.1. Splitting `tci.png` into Individual Observations

In `ssr/utils/infer_utils.py`:
- **Function**: `format_s2naip_data(s2_data, n_s2_images, device)`
- **Lines 8**:
  ```python
  s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))
  ```
- **Line 11**:
  ```python
  s2_image = s2_chunks[0]
  ```
The raw image array `s2_data` is read by `skimage.io.imread` from `tci.png`. Because `tci.png` is vertically stacked with shape $(1024, 32, 3)$ (or in PIL conventions $(W=32, H=1024)$), reshaping with $(-1, 32, 32, 3)$ segments the vertical column into $T = 1024 / 32 = 32$ individual temporal observations of shape $(32, 32, 3)$. The first frame `s2_chunks[0]` is extracted as `s2_image` for saving as `lr.png` reference.

In `run_full_validation_inference.py`:
- **Lines 175-177**:
  ```python
  image = skimage.io.imread(str(input_path))
  if image.ndim != 3 or image.shape[1] != 32 or image.shape[2] != 3 or image.shape[0] % 32:
      raise ValueError(f"Expected stacked RGB TCI shape (32*n,32,3), got {image.shape}")
  ```
This asserts strict conformance to $(32 \cdot T, 32, 3)$.

### 2.2. Determining Valid Observations

In `ssr/utils/infer_utils.py`:
- **Lines 15-20**:
  ```python
  goods, bads = [], []
  for i, ts in enumerate(s2_chunks):
      if [0, 0, 0] in ts:
          bads.append(i)
      else:
          goods.append(i)
  ```
The validity criterion is strictly whether any pixel in the $32 \times 32 \times 3$ chunk equals `[0, 0, 0]` (black nodata/border pixel). If any pixel is black, the entire frame is classified as `bads`; otherwise, it is classified as `goods`.

In `ssr/data/s2-naip_dataset.py` (lines 210-215) and `ssr/utils/data_utils.py` (lines 1-11):
- Training/dataset loading uses `has_black_pixels(ts)`:
  ```python
  def has_black_pixels(img_tensor):
      black_pixel = torch.tensor([0, 0, 0], dtype=img_tensor.dtype, device=img_tensor.device)
      is_black = torch.all(img_tensor == black_pixel[:, None, None], dim=0)
      return torch.any(is_black).item()
  ```
Both implementations enforce the identical nodata heuristic: any black pixel marks the observation invalid.

### 2.3. Selecting and Ordering the Eight Observations

In `ssr/utils/infer_utils.py`:
- **Lines 24-29**:
  ```python
  if len(goods) >= n_s2_images:
      rand_indices = random.sample(goods, n_s2_images)
  else:
      need = n_s2_images - len(goods)
      rand_indices = goods + random.sample(bads, need)
  ```
- **Line 30**:
  ```python
  s2_chunks = [s2_chunks[i] for i in rand_indices]
  ```
Critical characteristics of Python's `random.sample`:
1. `random.sample(population, k)` returns a list of $k$ items selected *without replacement*.
2. The returned list is in **random order**, not in sorted chronological or index order.
3. Because the order of `rand_indices` is preserved in `s2_chunks = [s2_chunks[i] for i in rand_indices]`, the ordering of temporal slices across the channel axis is determined by the permutation produced by `random.sample`.

### 2.4. Handling Missing or Invalid Frames

If `len(goods) < n_s2_images` (i.e. fewer than 8 valid frames):
- The pipeline consumes all available good frames (`goods`) and supplements them by randomly sampling `need` frames from `bads` (lines 27-28).
- If the total number of frames in the tile is less than 8 (`len(goods) + len(bads) < 8`), `random.sample(bads, need)` raises a `ValueError: Sample larger than population or negative`. In `small_val_set`, every tile has 32 frames, so this exception does not occur.

### 2.5. Normalization, Resizing, and Channel Concatenation

In `ssr/utils/infer_utils.py`:
- **Lines 34-36**:
  ```python
  s2_chunks = [torch.as_tensor(img).permute(2, 0, 1) for img in s2_chunks]
  s2_tensor = torch.cat(s2_chunks).unsqueeze(0)
  s2_tensor = s2_tensor.to(device).float() / 255
  ```
Step-by-step tensor tracing:
1. Each `img` is numpy uint8 $(32, 32, 3)$.
2. `.permute(2, 0, 1)` yields PyTorch tensor of shape $[3, 32, 32]$.
3. `torch.cat(s2_chunks)` concatenates the 8 tensors along dimension 0, resulting in a single tensor of shape $[24, 32, 32]$.
4. `.unsqueeze(0)` adds the batch dimension: $[1, 24, 32, 32]$.
5. `.to(device).float() / 255` casts to GPU float32 and scales pixel intensities to $[0.0, 1.0]$.
6. **No spatial resizing** is applied. Input remains native $32 \times 32$.

### 2.6. Converting Model Output to Saved Image Dimensions

In `run_full_validation_inference.py`:
- **Lines 200-212**:
  ```python
  with torch.inference_mode():
      prediction = model(input_tensor)
  torch.cuda.synchronize(device)
  prediction = torch.clamp(prediction, 0, 1).squeeze().cpu().numpy()
  if prediction.shape != (3, 128, 128):
      raise ValueError(f"Expected model output (3,128,128), got {prediction.shape}")
  prediction = np.transpose(prediction, (1, 2, 0))
  prediction = (prediction * 255).astype(np.uint8)
  ...
  skimage.io.imsave(str(Path(row["sr_output_path"])), prediction, check_contrast=False)
  ```
Step-by-step tensor tracing:
1. `model(input_tensor)` outputs tensor $[1, 3, 128, 128]$ via $4\times$ sub-pixel convolution / nearest upsample blocks.
2. `torch.clamp(prediction, 0, 1).squeeze().cpu().numpy()` clamps negative values and overflow, drops batch dimension, and moves to host NumPy array of shape $(3, 128, 128)$.
3. `.transpose(1, 2, 0)` converts planar RGB $[C, H, W]$ to interleaved RGB $[H, W, C]$, resulting in $(128, 128, 3)$.
4. Multiplying by 255 and casting to uint8 quantizes floats to $[0..255]$.
5. Output is saved via `skimage.io.imsave` as a standard 24-bit RGB PNG file.

---

## 3. Dataset Audit Findings

A complete programmatic census was executed against `data/small_val_set`:

| Metric | Inspected Value | Evidence / Command |
| :--- | :--- | :--- |
| **Matching S2 and NAIP Tiles** | Exactly 256 tiles | Both `sentinel2/` and `naip/` contain the identical set of 256 tile directory names. |
| **S2 Images per Tile** | 10 band files (b01, b05-b12, tci) | 2,560 Sentinel-2 PNGs total. |
| **NAIP Images per Tile** | 1 rgb.png | 256 NAIP PNGs total. |
| **Frames per `tci.png`** | Exactly 32 vertical sub-frames | Shape is $(1024, 32, 3) \rightarrow 1024 / 32 = 32$ for all 256 tiles. |
| **Valid Observations per Tile** | Exactly 32 valid frames (`goods=32`) | Tested for `[0,0,0] in ts` across all $256 \times 32 = 8,192$ sub-frames; 0 bad frames detected. |
| **Invalid Observations per Tile** | 0 invalid frames (`bads=0`) | 100% of sub-frames are free from black nodata pixels. |
| **Temporal Dates Available** | None | No EXIF timestamps, JSON sidecars, or filenames contain acquisition dates. |
| **Quality / Cloud Metadata** | None | No QA masks, cloud fraction metadata, or orbit tables are bundled. |

---

## 4. Determinism and Reproducibility Analysis

In `run_full_validation_inference.py`:
- `random.seed(seed)` (where `seed = 0`) is invoked at line 165 **once** prior to the loop over `work`.
- `work` is ordered alphabetically by `tile_id`:
  ```python
  rows = [{"tile_id": tile, ...} for tile in sorted(tiles)]
  ```
- Because all tiles have `goods = list(range(32))` and `bads = []`, the pseudorandom sequence of indices selected for tile $k$ is the $k$-th block of 8 numbers generated by Python's Mersenne Twister engine.
- **Consequence 1 (Global Determinism)**: A fresh execution from tile 0 to tile 255 will produce the exact same frame indices every time.
- **Consequence 2 (Order Dependency)**: If a single tile is run in isolation without running previous tiles, or if tiles are shuffled, the PRNG state will differ, producing different frame selections.
- **Consequence 3 (Uncontrolled Order)**: The ordering of the 8 selected frames within `rand_indices` is also random. Tile 0 (`20771_46525`) received indices `[17, 23, 22, 10, 27, 26, 12, 18]` in baseline run 1.

---

## 5. Architectural Sensitivity & Confounders

The Satlas-SR generator network is `SSR_RRDBNet` (defined in `ssr/archs/ssr_rrdbnet_arch.py`):
- `num_in_ch`: 24
- `conv_first`: `nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)`
- `body`: 23 Residual-in-Residual Dense Blocks (RRDB)
- `conv_up1`, `conv_up2`: Nearest-neighbor interpolation with $3 \times 3$ convolutions for $4\times$ spatial scaling.

### Architectural Implications:
1. **Lack of Permutation Invariance**:
   Unlike temporal attention networks or recurrent architectures (e.g. 3D-CNNs or temporal transformers) that treat temporal observations with permutation equivariance or explicit temporal positional embeddings, `SSR_RRDBNet` treats the 24 channels as 8 concatenated RGB triplets:
   $$\text{Channels } 0..2 = \text{Frame}_0, \quad 3..5 = \text{Frame}_1, \quad \dots, \quad 21..23 = \text{Frame}_7$$
   The weights of `conv_first.weight` have shape $[64, 24, 3, 3]$. Each channel index has its own specific learned convolution kernel. 
   Therefore:
   - Permuting the exact same 8 frames changes the initial feature activations!
   - Selecting different frames changes both the visual content and the channel positions.
2. **Training Bias**:
   The model checkpoint `esrgan_8S2.pth` was trained with `random.sample(goods, 8)` during every training iteration. The weights of `conv_first` learned to aggregate across arbitrary subsets and random permutations of Sentinel-2 observations.
   Consequently, imposing an artificial ordering (e.g. strictly sorted ascending indices) could introduce an out-of-distribution feature shift if the training distribution never saw strictly sorted inputs, or it might improve consistency. This is a critical hypothesis to test.

---

## 6. Construction of Alternative Frame Selections Without Invented Metadata

Because local timestamps and cloud masks are unavailable, we cannot construct "chronological" or "clearest" selections based on external metadata. However, valid, mathematically well-grounded alternative strategies can be constructed using only verified local data:

1. **Stratified / Strided Deterministic Selection**:
   - Uniform temporal coverage across the 32-frame buffer:
     $$\text{Indices} = [0, 4, 8, 12, 16, 20, 24, 28]$$
   - Captures evenly spaced observations across the temporal acquisition window without randomness.
2. **Positional Windowing Baselines**:
   - Early window: First 8 frames $[0, 1, 2, 3, 4, 5, 6, 7]$.
   - Late window: Last 8 frames $[24, 25, 26, 27, 28, 29, 30, 31]$.
   - Tests whether frame index proximity within `tci.png` correlates with seasonal or temporal clustering.
3. **Alternative Fixed Seeds (Seed Robustness)**:
   - Seed 42, Seed 123 (with identical PRNG sampling protocol).
   - Tests the sensitivity of the existing baseline to the specific random draw of seed 0.
4. **Sorted vs. Unsorted Permutation Test**:
   - Take the exact 8 frames chosen by baseline Seed 0 and sort them in ascending index order:
     $$\text{sorted}(\text{Seed 0 indices})$$
   - Isolate the pure effect of channel permutation while holding the frame content perfectly constant.
5. **Content-Derived Quality Heuristic (Zero Metadata Assumption)**:
   - Rank frames by image statistics:
     - *Mean Brightness / Cloud Avoidance*: Sentinel-2 clouds exhibit extreme reflectance across all RGB channels ($>200$). Selecting the 8 frames with lowest mean luminance or highest contrast tests whether cloud-filtering heuristics improve super-resolution quality.
