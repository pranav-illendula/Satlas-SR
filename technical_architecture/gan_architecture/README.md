# GAN Architecture: Generator, Discriminator & Objective Dynamics

This folder contains the complete Generative Adversarial Network (GAN) architecture suite for multi-frame satellite super-resolution.

---

## 1. Generator Architecture (`SSR_RRDBNet`)
- **Backbone**: 23 Residual-in-Residual Dense Blocks (RRDB) with residual scaling ($\beta = 0.2$).
- **Parameters**: 16,710,083 weights in the official 8-S2 model (`experiments/esrgan_8S2.pth`).
- **Input Representation**: Accepts $T$ temporal Sentinel-2 frames stacked along the channel axis. For 8 temporal revisits of 3 RGB bands, input shape is $[B, 24, 32, 32]$.
- **Upsampling Head**: Dual $2\times$ nearest-neighbor interpolations interleaved with $3\times 3$ convolutions and LeakyReLU ($\alpha = 0.2$), yielding a $4\times$ spatial upscale to $128 \times 128$ px (2.5m ground resolution).

---

## 2. U-Net Discriminator with Spectral Normalization (`SSR_UNetDiscriminatorSN`)
- **Structural Design**: 3 downsampling encoder stages with stride 2 and 3 upsampling decoder stages with bilateral skip connections.
- **Spectral Normalization**: Applied to all internal convolutional weights via `torch.nn.utils.spectral_norm`. Enforces a Lipschitz constraint:
  $$\|D(x_1) - D(x_2)\| \le K \|x_1 - x_2\|$$
  preventing mode collapse and gradient explosion during adversarial optimization.
- **Pixel-Level Feedback**: Rather than collapsing features to a scalar real/fake score, the output is a full $128 \times 128$ spatial map, evaluating structural fidelity at every spatial location.

---

## 3. Relativistic Average GAN (RaGAN) Formulation
Instead of assessing whether an image is unconditionally real or fake, the Relativistic discriminator estimates the probability that a real image $x_r$ is relatively more realistic than an average fake image $x_f$:
$$D_{Ra}(x_r, x_f) = \sigma\left(C(x_r) - \mathbb{E}_{x_f}[C(x_f)]\right)$$
$$D_{Ra}(x_f, x_r) = \sigma\left(C(x_f) - \mathbb{E}_{x_r}[C(x_r)]\right)$$

### Generator Adversarial Loss:
$$L_G^{Ra} = -\mathbb{E}_{x_r}\left[\log(1 - D_{Ra}(x_r, x_f))\right] - \mathbb{E}_{x_f}\left[\log(D_{Ra}(x_f, x_r))\right]$$

### Composite Objective Function:
$$L_{\text{total}} = L_{\text{pix}}(G(x), y) + \lambda_{\text{percep}} L_{\text{percep}}(G(x), y) + \lambda_{\text{gan}} L_G^{Ra}$$
where:
- $L_{\text{pix}}$: L1 pixel distance between generated and NAIP ground truth.
- $L_{\text{percep}}$: Feature distance in VGG-19 feature activation space.
- $L_G^{Ra}$: Relativistic adversarial score.

