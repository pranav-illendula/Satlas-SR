# DeepSat-SR: Comprehensive Technical Approach & Implementation Deep Dive

This technical specification provides an exhaustive, code-level breakdown of the **DeepSat-SR** multi-temporal satellite super-resolution system. It details the end-to-end engineering pipeline: from source data provenance and tensor ingestion to the 23-block residual-in-residual generator, relativistic U-Net discriminator, mathematical loss formulation, and final predicted image synthesis.

---

## 1. Data Origin, Curation & Input Engineering

### 1.1 Source Imagery & Physical Constellations
The dataset pairs two distinct optical Earth observation platforms across the contiguous United States (imagery collected between 2019 and 2020):

1. **Low-Resolution Satellite Ingestion (Sentinel-2 L1C):**
   - **Operator:** European Space Agency (ESA) Copernicus Constellation.
   - **Ground Sample Distance (GSD):** $10\,\text{m}$ per pixel (for visible RGB bands).
   - **Radiometric Format:** Level-1C top-of-atmosphere reflectance. True Color Image (TCI) tiles are 8-bit normalized images ($0 - 255$ integer scale). Non-TCI multi-spectral bands (e.g., NIR, RedEdge) are 16-bit integers divided by $8160$ and clipped to $[0.0, 1.0]$.
   - **Revisit Frequency:** Global pass every 2–5 days.
2. **High-Resolution Ground Truth Target (USDA NAIP):**
   - **Operator:** United States Department of Agriculture (USDA) National Agriculture Imagery Program.
   - **Platform:** High-altitude airborne orthorectified aerial photography.
   - **Native Resolution:** $0.6\,\text{m} - 1.0\,\text{m}$ GSD.
   - **Downsampled Reference Target:** Subsampled by a factor of 4 to exactly $2.5\,\text{m}$ GSD to maintain a consistent $4\times$ super-resolution scaling factor relative to Sentinel-2 ($10\,\text{m} \to 2.5\,\text{m}$).
   - **Image Dimensions:** $128 \times 128$ pixels, 3 channels (RGB), $[B, 3, 128, 128]$.

---

### 1.2 Geographic Tiling & The Web-Mercator Grid
Both datasets adhere to a unified Web-Mercator grid hierarchy:
- **Tiling Grid:** Discretized at Zoom Level 17 ($2^{17} \times 2^{17}$ global grid cells, tile identifier format `XXXXX_YYYYY`).
- **Spatial Alignment:** A high-resolution NAIP tile of size $128 \times 128$ pixels at $2.5\,\text{m}$ GSD corresponds precisely to a $32 \times 32$ pixel footprint in Sentinel-2 at $10\,\text{m}$ GSD.
- **Dataset Partitions:**
  - `train_urban_set` (S2-NAIP): $\sim 1.1 \text{ million}$ paired scenes within a $5\,\text{km}$ radius of U.S. cities with populations $\ge 50,\!000$. Urban imagery exhibits dense high-frequency edges (rooftops, roads, ports) essential for training super-resolution priors.
  - `small_val_set`: $256$ representative urban tiles used for rigorous standardized evaluation.

---

### 1.3 The 8 Multi-Temporal Input Images
Rather than single-image super-resolution (SISR), DeepSat-SR implements **Multi-Image Super-Resolution (MISR)**:

```
 Sentinel-2 Time Series (Overlapping within 3 Months of NAIP acquisition)
 ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        ┌───────────────┐
 │ Acquisition 1 │ │ Acquisition 2 │ │ Acquisition 3 │  ...   │ Acquisition 8 │
 │   [3, 32, 32] │ │   [3, 32, 32] │ │   [3, 32, 32] │        │   [3, 32, 32] │
 └───────┬───────┘ └───────┬───────┘ └───────┬───────┘        └───────┬───────┘
         │                 │                 │                        │
         └─────────────────┴────────┬────────┴────────────────────────┘
                                    ▼
      Concatenation across channels: [ 8 x 3, 32, 32 ] = [ 24, 32, 32 ]
```

1. **Temporal Selection Window:** For each NAIP ground truth chip, candidate Sentinel-2 scenes are restricted to acquisitions within **$\pm 3$ months** of the NAIP flight date.
2. **Quality & Cloud-Haze Filtering:** The dataloader (`S2NAIPDataset` in `ssr/data/s2-naip_dataset.py`) inspects every candidate $32 \times 32$ tile with `has_black_pixels()`. Tiles suffering from nodata boundaries, missing sensors, or extreme shadow drops are culled.
3. **Temporal Invariance:** Chronological order is non-critical; 8 valid acquisitions are randomly sampled from the candidate pool during training to ensure temporal permutation robustness.
4. **Input Tensor Construction:**
   - Raw byte values ($[0, 255]$) are converted to 32-bit floating point and normalized:
     $$X_{\text{norm}} = \frac{X}{255.0} \in [0.0, 1.0]$$
   - The 8 separate RGB images (each $[3, 32, 32]$) are concatenated along the channel axis:
     $$\mathbf{X}_{\text{in}} \in \mathbb{R}^{B \times 24 \times 32 \times 32}$$
   - Channels $0-2$: Frame 1 (R, G, B), Channels $3-5$: Frame 2 (R, G, B), $\dots$, Channels $21-23$: Frame 8 (R, G, B).

---

## 2. Generator Micro-Architecture: 23-Block SSR_RRDBNet

The generator adapts the **RRDBNet (Residual-in-Residual Dense Block Network)** backbone (`ssr/archs/rrdbnet_arch.py`), totaling **$16,710,083$ learnable parameters**.

```
+---------------------------------------------------------------------------------------------------+
|                                  SSR_RRDBNet GENERATOR TOPOLOGY                                   |
+---------------------------------------------------------------------------------------------------+
 Input Tensor: [B, 24, 32, 32] (8 temporal Sentinel-2 passes x 3 RGB channels)
   │
   ▼
 [ conv_first: Conv2d(24 -> 64, k=3, s=1, p=1) ] ─────────────────────────────────┐ (Long Skip)
   │                                                                               │
   ▼ Feat Map: [B, 64, 32, 32]                                                     │
 ┌───────────────────────────────────────────────────────────────────────────────┐ │
 │ 23 x Cascaded RRDB Modules (Residual in Residual Dense Blocks)                │ │
 │                                                                               │ │
 │   ┌─────────────────────────────────────────────────────────────────────────┐ │ │
 │   │ Single RRDB Unit:                                                       │ │ │
 │   │                                                                         │ │ │
 │   │   x_in ──► [ RDB 1 ] ──► [ RDB 2 ] ──► [ RDB 3 ] ──( * 0.2 )──(+)       │ │ │
 │   │     │                                                     ▲             │ │ │
 │   │     └─────────────────────────────────────────────────────┘             │ │ │
 │   │     Output = RDB3_out * 0.2 + x_in                                      │ │ │
 │   │                                                                         │ │ │
 │   │   * Inside Each RDB (Residual Dense Block):                             │ │ │
 │   │     x0 = Input [64 channels]                                            │ │ │
 │   │     x1 = LeakyReLU( Conv2d(64 -> 32, k=3, s=1, p=1)(x0) )               │ │ │
 │   │     x2 = LeakyReLU( Conv2d(96 -> 32, k=3, s=1, p=1)([x0, x1]) )        │ │ │
 │   │     x3 = LeakyReLU( Conv2d(128 -> 32, k=3, s=1, p=1)([x0, x1, x2]) )    │ │ │
 │   │     x4 = LeakyReLU( Conv2d(160 -> 32, k=3, s=1, p=1)([x0..x3]) )       │ │ │
 │   │     x5 = Conv2d(192 -> 64, k=3, s=1, p=1)([x0..x4])                    │ │ │
 │   │     RDB_out = x5 * 0.2 + x0                                             │ │ │
 │   └─────────────────────────────────────────────────────────────────────────┘ │ │
 └───────────────────────────────────────────────────────────────────────────────┘ │
   │                                                                               │
   ▼                                                                               │
 [ conv_body: Conv2d(64 -> 64, k=3, s=1, p=1) ]                                    │
   │                                                                               │
   ▼                                                                               │
 [ Element-wise Addition: feat + body_feat ] ◄─────────────────────────────────────┘
   │
   ▼ Shape: [B, 64, 32, 32]
 [ Nearest-Neighbor 2x Interpolation ] ──► [ conv_up1: Conv2d(64 -> 64) ] ──► [ LReLU ] ──► [B, 64, 64, 64]
   │
   ▼
 [ Nearest-Neighbor 2x Interpolation ] ──► [ conv_up2: Conv2d(64 -> 64) ] ──► [ LReLU ] ──► [B, 64, 128, 128]
   │
   ▼
 [ conv_hr: Conv2d(64 -> 64, k=3, s=1, p=1) ] ──► [ LeakyReLU(0.2) ]
   │
   ▼
 [ conv_last: Conv2d(64 -> 3, k=3, s=1, p=1) ] ──► Output Super-Resolved Image: [B, 3, 128, 128]
```

### 2.1 First Convolution (`conv_first`)
- **Kernel Size:** $3 \times 3$, stride 1, padding 1.
- **Input Channels:** $24$ ($8$ images $\times 3$ RGB channels).
- **Output Channels:** $64$ base intermediate feature maps.
- **Role:** Projects multi-temporal channel concatenation into high-dimensional feature space without reducing spatial dimensions ($32 \times 32$).

### 2.2 Deep Trunk: 23 Cascaded RRDB Blocks
The core trunk consists of 23 sequentially connected RRDB units. An RRDB employs a **three-tier hierarchical residual architecture**:

1. **Growth Rate ($G = 32$):** Within each dense block, every convolutional layer outputs 32 channels.
2. **Dense Channel Concatenation:** Each layer within an RDB receives the concatenated outputs of all preceding layers:
   - Layer 1 receives $64$ ch $\to$ outputs $32$ ch.
   - Layer 2 receives $64 + 32 = 96$ ch $\to$ outputs $32$ ch.
   - Layer 3 receives $96 + 32 = 128$ ch $\to$ outputs $32$ ch.
   - Layer 4 receives $128 + 32 = 160$ ch $\to$ outputs $32$ ch.
   - Layer 5 receives $160 + 32 = 192$ ch $\to$ outputs $64$ ch (bottleneck compression).
3. **Residual Scaling ($\alpha = 0.2$):** Residual additions at both the RDB level and RRDB level are multiplied by $0.2$:
   $$\mathbf{x}_{\text{RDB\_out}} = \mathbf{x}_0 + 0.2 \cdot \mathbf{x}_5$$
   $$\mathbf{x}_{\text{RRDB\_out}} = \mathbf{x}_{\text{in}} + 0.2 \cdot \text{RDB}_3(\text{RDB}_2(\text{RDB}_1(\mathbf{x}_{\text{in}})))$$
   *Mathematical Rationale:* As network depth exceeds 100 total convolution layers, unscaled residual additions cause training numerical instability and gradient explosion. A scaling factor of $0.2$ prevents variance explosion across deep paths.
4. **Global Trunk Skip Connection:** The output of the 23rd RRDB passes through `conv_body` ($64 \to 64$, $3 \times 3$) and adds directly back to the initial feature representation (`feat + body_feat`). This allows the trunk to strictly learn high-frequency residual differences while preserving the low-frequency background.

---

### 2.3 Progressive $4\times$ Upsampling Pipeline
Rather than performing a single $4\times$ upsampling step (which creates checkerboard deconvolution artifacts), the network upsamples progressively in two $2\times$ stages:

1. **Upsample Stage 1 ($32 \times 32 \to 64 \times 64$):**
   - Nearest-neighbor interpolation with `scale_factor=2`.
   - Convolution `conv_up1`: $3 \times 3$ Conv ($64 \to 64$, stride 1, padding 1).
   - Activation: LeakyReLU ($\text{negative\_slope}=0.2$).
2. **Upsample Stage 2 ($64 \times 64 \to 128 \times 128$):**
   - Nearest-neighbor interpolation with `scale_factor=2`.
   - Convolution `conv_up2`: $3 \times 3$ Conv ($64 \to 64$, stride 1, padding 1).
   - Activation: LeakyReLU ($\text{negative\_slope}=0.2$).
3. **High-Resolution Feature Refinement (`conv_hr`):**
   - $3 \times 3$ Conv ($64 \to 64$, stride 1, padding 1) + LeakyReLU($0.2$). Blends upsampled features and eliminates interpolation seams.
4. **Final Output Projection (`conv_last`):**
   - $3 \times 3$ Conv ($64 \to 3$, stride 1, padding 1).
   - **Activation:** Linear (identity). No clipping or sigmoid is placed on the final layer, allowing direct regression of continuous RGB radiance.

---

## 3. Discriminator Architecture: Spectral-Normalized U-Net

The adversarial discriminator (`SSR_UNetDiscriminatorSN` in `ssr/archs/discriminator_arch.py`) uses a **U-Net architecture with Spectral Normalization**:

```
+---------------------------------------------------------------------------------------------------+
|                        SSR_UNetDiscriminatorSN (SPECTRAL NORMALIZATION)                           |
+---------------------------------------------------------------------------------------------------+
 Input Image: [B, 3, 128, 128] (Real NAIP or Super-Resolved Output)
   │
   ▼
 [ conv0: Conv2d(3 -> 64, k=3, s=1, p=1) ] ──► x0: [B, 64, 128, 128] ─────────────┐ (Skip 0)
   │ (Downsample)                                                                  │
   ▼                                                                               │
 [ conv1: SN(Conv2d(64 -> 128, k=4, s=2, p=1)) ] ──► x1: [B, 128, 64, 64] ──────┐ │ (Skip 1)
   │ (Downsample)                                                               │ │
   ▼                                                                            │ │
 [ conv2: SN(Conv2d(128 -> 256, k=4, s=2, p=1)) ] ──► x2: [B, 256, 32, 32] ───┐ │ │ (Skip 2)
   │ (Downsample)                                                            │ │ │
   ▼                                                                         │ │ │
 [ conv3: SN(Conv2d(256 -> 512, k=4, s=2, p=1)) ] ──► x3: [B, 512, 16, 16]   │ │ │
   │                                                                         │ │ │
   ▼ (Bilinear Interpolate 2x -> 32x32)                                      │ │ │
 [ conv4: SN(Conv2d(512 -> 256, k=3, s=1, p=1)) ]                            │ │ │
   │                                                                         │ │ │
   ▼ Element-wise Addition with Skip 2: [ x4 + x2 ] ◄────────────────────────┘ │ │
   │ (Bilinear Interpolate 2x -> 64x64)                                        │ │
   ▼                                                                           │ │
 [ conv5: SN(Conv2d(256 -> 128, k=3, s=1, p=1)) ]                            │ │
   │                                                                           │ │
   ▼ Element-wise Addition with Skip 1: [ x5 + x1 ] ◄──────────────────────────┘ │
   │ (Bilinear Interpolate 2x -> 128x128)                                        │
   ▼                                                                             │
 [ conv6: SN(Conv2d(128 -> 64, k=3, s=1, p=1)) ]                                │
   │                                                                             │
   ▼ Element-wise Addition with Skip 0: [ x6 + x0 ] ◄────────────────────────────┘
   │
   ▼
 [ conv7: SN(Conv2d(64 -> 64, k=3, s=1, p=1)) ] ──► LeakyReLU(0.2)
   │
   ▼
 [ conv8: SN(Conv2d(64 -> 64, k=3, s=1, p=1)) ] ──► LeakyReLU(0.2)
   │
   ▼
 [ conv9: Conv2d(64 -> 1, k=3, s=1, p=1) ] ──► Output Real/Fake Map: [B, 1, 128, 128]
```

### 3.1 Why a U-Net Discriminator over Standard PatchGAN?
1. **Dense Pixel-Wise Supervision:** Standard discriminators (e.g., VGG style) reduce the entire image to a single scalar or coarse $16 \times 16$ grid. The U-Net discriminator outputs a full-resolution **$128 \times 128$ score matrix**, grading every individual building corner, road edge, and tree texture.
2. **Spectral Normalization (`spectral_norm`):** Every intermediate convolution is constrained by dividing weight matrices by their spectral norm (largest singular value $\sigma(W)$):
   $$\bar{W}_{\text{SN}} = \frac{W}{\sigma(W)}$$
   This enforces a 1-Lipschitz continuity constraint, eliminating discriminator gradient explosion and stabilizing GAN oscillations.

---

## 4. Multi-Task Training Objectives & Loss Formulations

During optimization (`ssr/models/ssr_esrgan_model.py`), the generator objective balances three complementary loss functions:

$$\mathcal{L}_{\text{total}}^G = \mathcal{L}_1 + \lambda_{\text{percep}} \mathcal{L}_{\text{percep}} + \lambda_{\text{adv}} \mathcal{L}_{\text{RaGAN}}$$

### 4.1 Pixel Reconstruction Loss ($\mathcal{L}_1$)
Computes the mean absolute error between super-resolved output $\hat{Y} = G(\mathbf{X}_{\text{in}})$ and ground truth NAIP target $Y$:

$$\mathcal{L}_1 = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^3 \sum_{i=1}^{128} \sum_{j=1}^{128} \left| \hat{Y}_{c,i,j} - Y_{c,i,j} \right|$$

- **Technical Advantage:** Compared to $L_2$ (Mean Squared Error), $L_1$ does not over-penalize large individual pixel disparities, preventing the network from settling on blurry, median-averaged color fields.

---

### 4.2 Deep Perceptual Feature Loss ($\mathcal{L}_{\text{percep}}$)
Measures distance between deep convolutional feature maps extracted from layer `conv3_4` of an ImageNet-pretrained **VGG-16** network:

$$\mathcal{L}_{\text{percep}} = \frac{1}{C_l H_l W_l} \sum_{c=1}^{C_l} \sum_{i=1}^{H_l} \sum_{j=1}^{W_l} \left( \phi_{\text{conv3\_4}}(\hat{Y})_{c,i,j} - \phi_{\text{conv3\_4}}(Y)_{c,i,j} \right)^2$$

- **Technical Advantage:** Pure pixel loss yields high PSNR but perceptually blurry textures. Perceptual loss enforces high-level structural semantics, natural contrast, and spatial edge fidelity.
- **Loss Weight:** $\lambda_{\text{percep}} = 1.0$.

---

### 4.3 Relativistic Average GAN Loss ($\mathcal{L}_{\text{RaGAN}}$)
Standard GAN discriminators estimate the absolute probability that an input is real ($D(x) = \sigma(C(x))$). In contrast, the **Relativistic Average Discriminator** calculates the relative probability that a real image $x_r$ is more realistic than an average generated image $x_f$:

$$D_{\text{Ra}}(x_r, x_f) = \sigma\left( C(x_r) - \mathbb{E}_{x_f}[C(x_f)] \right)$$
$$D_{\text{Ra}}(x_f, x_r) = \sigma\left( C(x_f) - \mathbb{E}_{x_r}[C(x_r)] \right)$$

#### Discriminator Loss Function ($\mathcal{L}_D$):
$$\mathcal{L}_D = - \mathbb{E}_{x_r}\left[ \log\left( D_{\text{Ra}}(x_r, \hat{Y}) \right) \right] - \mathbb{E}_{\hat{Y}}\left[ \log\left( 1 - D_{\text{Ra}}(\hat{Y}, x_r) \right) \right]$$

#### Generator Adversarial Loss ($\mathcal{L}_{\text{adv}}^G$):
$$\mathcal{L}_{\text{adv}}^G = - \mathbb{E}_{x_r}\left[ \log\left( 1 - D_{\text{Ra}}(x_r, \hat{Y}) \right) \right] - \mathbb{E}_{\hat{Y}}\left[ \log\left( D_{\text{Ra}}(\hat{Y}, x_r) \right) \right]$$

- **Technical Advantage:** Relativistic formulation forces the generator to simultaneously sharpen fake textures while actively degrading unrealistic synthetic artifacts, speeding up convergence.
- **Loss Weight:** $\lambda_{\text{adv}} = 0.005$.

---

### 4.4 Target Preprocessing: Unsharp Masking (USM)
Prior to loss computation, ground truth NAIP targets are pre-processed using Gaussian **Unsharp Masking (USM)**:
$$Y_{\text{usm}} = Y + \lambda_{\text{usm}} \cdot (Y - \text{GaussianBlur}(Y, k=7, \sigma=1.5))$$
This sharpens boundary transitions in the reference targets, guiding the generator to synthesize crisp physical edges.

---

### 4.5 Training Dynamics & Exponential Moving Average (EMA)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.99$, learning rate $\eta = 10^{-4}$).
- **Exponential Moving Average:** Generator weights are tracked with EMA ($\text{decay} = 0.999$):
  $$\theta_{\text{EMA}}^{(t)} = 0.999 \cdot \theta_{\text{EMA}}^{(t-1)} + 0.001 \cdot \theta^{(t)}$$
  Inference uses the EMA weights, filtering out high-frequency parameter oscillations.

---

## 5. Output Synthesis, Tiling & Predicted Image Reconstruction

### 5.1 Output Tensor Dimensions & Physical Scaling
- **Input Dimensions:** $\mathbf{X}_{\text{in}} \in \mathbb{R}^{B \times 24 \times 32 \times 32}$ (representing an area of $320\,\text{m} \times 320\,\text{m}$ at $10\,\text{m}$ GSD).
- **Output Dimensions:** $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ (representing the identical $320\,\text{m} \times 320\,\text{m}$ footprint at $2.5\,\text{m}$ GSD).
- **Physical Sampling Multiplication:** A $4\times$ spatial increase per dimension ($16\times$ total pixel density increase from 1,024 pixels to 16,384 pixels).

---

### 5.2 Wide-Area Grid Inference & Stitching (`infer_grid.py`)
To super-resolve an entire Sentinel-2 regional scene:
1. The regional tile is broken down into a $16 \times 16$ grid of $32 \times 32$ chunks ($256$ chunks total).
2. The model infers each chunk independently, yielding $256$ outputs of size $128 \times 128$.
3. The chunks are stitched along coordinate boundaries into a single continuous **$2048 \times 2048$ pixel GeoTIFF composite**, representing an area of $\sim 5.12\,\text{km} \times 5.12\,\text{km}$.

---

## 6. Comprehensive Technical Parameter Summary

| Component | Layer / Module | Input Channels | Output Channels | Kernel / Stride | Activation | Learnable Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Input** | `S2NAIPDataset` | — | $24$ | — | — | — |
| **Generator** | `conv_first` | $24$ | $64$ | $3 \times 3$, s1, p1 | Identity | $13,\!888$ |
| **Generator** | 23 RRDB Trunk | $64$ | $64$ | Dense $3 \times 3$ | LeakyReLU(0.2) | $\sim 15,\!800,\!000$ |
| **Generator** | `conv_body` | $64$ | $64$ | $3 \times 3$, s1, p1 | Identity | $36,\!928$ |
| **Generator** | `conv_up1` ($2\times$) | $64$ | $64$ | $3 \times 3$, s1, p1 | LeakyReLU(0.2) | $36,\!928$ |
| **Generator** | `conv_up2` ($2\times$) | $64$ | $64$ | $3 \times 3$, s1, p1 | LeakyReLU(0.2) | $36,\!928$ |
| **Generator** | `conv_hr` | $64$ | $64$ | $3 \times 3$, s1, p1 | LeakyReLU(0.2) | $36,\!928$ |
| **Generator** | `conv_last` | $64$ | $3$ | $3 \times 3$, s1, p1 | Identity | $1,\!731$ |
| **Discriminator** | `conv0` | $3$ | $64$ | $3 \times 3$, s1, p1 | LeakyReLU(0.2) | $1,\!792$ |
| **Discriminator** | Encoder `conv1-3` | $64 \to 256$ | $128 \to 512$ | $4 \times 4$, s2, p1 | LeakyReLU(0.2) | $\sim 3,\!200,\!000$ |
| **Discriminator** | Decoder `conv4-6` | $512 \to 128$ | $256 \to 64$ | $3 \times 3$, s1, p1 | LeakyReLU(0.2) | $\sim 1,\!100,\!000$ |
| **Discriminator** | Output `conv7-9` | $64$ | $1$ | $3 \times 3$, s1, p1 | Linear | $75,\!000$ |
| **Total Model** | **Generator + Discriminator** | — | — | — | — | **$\sim 21.0 \text{ Million}$** |
