# Residual & Dense Architectural Blocks

This folder contains the core convolutional building blocks that power the Satlas-SR super-resolution generators and feature extraction backbones.

---

## 1. Residual Dense Block (`ResidualDenseBlock`)
The **Residual Dense Block (RDB)** combines multi-layer dense connections with local residual learning.
Inside each RDB:
- 5 convolutional layers are sequentially applied.
- The input to layer $l$ is the channel concatenation of the block's input $x_0$ and the activations of all preceding layers:
  $$x_l = \sigma\left(W_l * [x_0, x_1, \dots, x_{l-1}]\right), \quad l \in \{1, 2, 3, 4\}$$
- The final $1\times 1$ convolution aggregates all accumulated feature channels back to `num_feat` (64 channels):
  $$x_5 = W_5 * [x_0, x_1, x_2, x_3, x_4]$$
- A residual scale multiplier $\beta = 0.2$ stabilizes backpropagation through deep cascades:
  $$x_{\text{out}} = x_0 + \beta \cdot x_5$$

---

## 2. Residual in Residual Dense Block (`RRDB`)
The **RRDB** introduces higher-level residual learning by stacking three `ResidualDenseBlock` instances inside an outer residual skip connection:
$$y = x + \beta \cdot \text{RDB}_3\left(\text{RDB}_2\left(\text{RDB}_1(x)\right)\right)$$

### Key Advantages:
1. **Gradient Flow**: Gradients can propagate unimpeded across both inner dense paths and outer residual highways.
2. **Deep Trunk Scalability**: Enables scaling the ESRGAN generator to 23 blocks (over 16.7 million parameters) without optimization collapse.
3. **No Batch Normalization**: Standard Batch Normalization layers are intentionally excluded (`ResidualBlockNoBN`). In super-resolution tasks, BN normalizes batch statistics across diverse images, which flattens local dynamic ranges and creates unpleasant high-frequency halos around sharp edges.

---

## 3. Spatio-Temporal & Upsampling Modules
- **`FusionBlock` & `RecursiveFusion`**: Implements 2D temporal aggregation across multiple Sentinel-2 revisits (up to 16 temporal captures).
- **`PixelShuffleBlock`**: Implements efficient sub-pixel convolution upsampling ($O(1)$ runtime relative to transposed convolution).
- **`Upsample`**: Nearest-neighbor interpolation combined with convolutional refinement, eliminating checkerboard interpolation artifacts.

