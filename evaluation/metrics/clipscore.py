"""
CLIPScore metric for high-level semantic alignment in remote sensing.
Computes cosine similarity in vision-language feature space.
"""
from __future__ import annotations

import numpy as np


def calculate_clipscore(
    img: np.ndarray,
    img2: np.ndarray,
    clip_model: str = "clipa-ViT-bigG-14",
    device: str = "cuda",
) -> float:
    """
    Computes CLIP feature cosine similarity between prediction and reference.
    Official Satlas-SR benchmark uses OpenCLIP ViT-bigG-14-CLIPA-336 pretrained on datacomp1b.
    Output is bounded in [0.70, 1.00] for natural satellite scenes.
    """
    try:
        import torch
        import torch.nn.functional as F
        import open_clip
    except ImportError:
        raise ImportError("open_clip_torch and torch are required for CLIPScore. Install via: pip install open_clip_torch")

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    if clip_model == "clipa-ViT-bigG-14":
        model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14-CLIPA-336", pretrained="datacomp1b")
        img_size = (336, 336)
    elif clip_model == "siglip-ViT-SO400M-14":
        model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
        img_size = (384, 384)
    else:
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        img_size = (224, 224)

    model = model.to(dev).eval()

    t1 = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).to(dev).float() / 255.0
    t2 = torch.as_tensor(img2).permute(2, 0, 1).unsqueeze(0).to(dev).float() / 255.0

    t1 = F.interpolate(t1, img_size, mode="bilinear", align_corners=False)
    t2 = F.interpolate(t2, img_size, mode="bilinear", align_corners=False)

    with torch.no_grad():
        feats1 = model.encode_image(t1)
        feats2 = model.encode_image(t2)
        score = F.cosine_similarity(feats1, feats2).detach().item()

    return float(score)

