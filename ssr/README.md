# Satlas Super-Resolution (`ssr`) Engine

`ssr/` is the core modeling, data loading, training, and inference engine for Satlas-SR.

## Package Architecture

```
ssr/
├── train.py          # Distributed training entry point (PyTorch DDP)
├── test.py           # Testing runner with quantitative metric logging
├── infer.py          # Single-tile inference script without ground truth
├── infer_grid.py     # 16x16 contiguous tile grid inference and mosaic stitching
├── archs/            # Model architecture classes (RRDBNet, HighResNet, SRCNN)
├── data/             # PyTorch Dataset loaders (S2NAIP, OLI2MSI, WorldStrat)
├── losses/           # Specialized loss functions (CLIP semantic loss, SSIM loss)
├── metrics/          # Quantitative metric functions (cPSNR, LPIPS, CLIPScore)
├── models/           # Training orchestrators (SSRESRGANModel)
├── options/          # YAML configuration templates for training and inference
└── utils/            # Model builders, option parsers, and data formatting
```

## Core Modules & Functionality

| Module / Script | Description |
| :--- | :--- |
| `train.py` | Main training pipeline supporting single-GPU execution and distributed multi-GPU training via `torch.distributed`. |
| `test.py` | Full validation/test runner that computes cPSNR, PSNR, SSIM, LPIPS, and CLIPScore when ground-truth aerial imagery is available. |
| `infer.py` | Standalone inference script that loads raw Sentinel-2 multi-frame inputs and outputs super-resolved PNGs. |
| `infer_grid.py` | High-throughput mosaic runner that processes an entire $16 \times 16$ tile grid (256 tiles) and stitches them into a seamless high-resolution mosaic. |
| `archs/` | Contains neural network definitions, including `rrdbnet_arch.py` (ESRGAN generator), `srcnn_arch.py`, and `highresnet_arch.py`. |
| `data/` | Implements dataset loaders: `s2naip_dataset.py` (temporal stacks paired with NAIP chips), `oli2msi_dataset.py`, and `worldstrat_dataset.py`. |
| `losses/` | Contains custom loss functions: `clip_loss.py` (semantic cosine loss) and `ssim_loss.py`. |
| `models/` | Houses the `SSRESRGANModel` class orchestrating optimizers, learning rate schedulers, and adversarial step dynamics. |
| `options/` | YAML configuration files defining hyper-parameters, network architectures, and dataset directories. |
| `utils/` | Utility functions for option parsing (`options.py`), model instantiation (`model_utils.py`), and frame formatting (`infer_utils.py`). |

