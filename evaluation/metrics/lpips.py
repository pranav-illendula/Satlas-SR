"""
Learned Perceptual Image Patch Similarity (LPIPS) metric.
Measures perceptual patch similarity using deep features (Zhang et al., 2018).
"""
from __future__ import annotations

import numpy as np


def calculate_lpips(
    img: np.ndarray,
    img2: np.ndarray,
    lpips_model: str = "vgg",
    device: str = "cuda",
) -> float:
    """
    Computes LPIPS distance between two RGB images [0, 255].
    Lower score indicates higher perceptual similarity.
    """
    try:
        import lpips
        import torch
    except ImportError:
        raise ImportError("lpips and torch are required for LPIPS evaluation. Install via: pip install lpips")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    net = "vgg" if lpips_model.lower() == "vgg" else "alex"
    loss_fn = lpips.LPIPS(net=net).to(dev)

    t1 = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).to(dev).float() / 255.0
    t2 = torch.as_tensor(img2).permute(2, 0, 1).unsqueeze(0).to(dev).float() / 255.0

    # LPIPS expects input scaled to [-1, 1]
    t1 = t1 * 2.0 - 1.0
    t2 = t2 * 2.0 - 1.0

    with torch.no_grad():
        score = loss_fn(t1, t2).detach().item()
    return float(score)

