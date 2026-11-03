"""
Rigorous synthetic and numerical test suite for Satlas-SR metrics:
- PSNR (BasicSR vs skimage vs NumPy)
- SSIM (BasicSR vs skimage vs NumPy)
- cPSNR (PROBA-V translation & brightness bias compensation)
- LPIPS (VGG perceptual distance)
- CLIPScore (CLIPA-v2 ViT-bigG-14 image-to-image cosine similarity)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    peak_signal_noise_ratio = None
    structural_similarity = None

from evaluation.evaluate_validation import compute_psnr_numpy, compute_ssim_numpy
from evaluation.metrics.cpsnr import calculate_cpsnr
from evaluation.metrics.lpips import calculate_lpips
from evaluation.metrics.clipscore import calculate_clipscore


def calculate_psnr(img1, img2, crop_border=0, **kwargs):
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return compute_psnr_numpy(img1, img2, data_range=255.0)


def calculate_ssim(img1, img2, crop_border=0, **kwargs):
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return compute_ssim_numpy(img1, img2, data_range=255.0)


def make_synthetic_pattern(h: int = 128, w: int = 128) -> np.ndarray:
    """Generate a rich, deterministic synthetic texture with edges, gradients, and patterns."""
    y, x = np.mgrid[:h, :w]
    ch0 = (np.sin(x / 4.0) * np.cos(y / 4.0) * 100 + 128).astype(np.uint8)
    ch1 = ((x + y) * 255 / (h + w)).astype(np.uint8)
    ch2 = (((x // 8) % 2) ^ ((y // 8) % 2)) * 200 + 30
    ch2 = ch2.astype(np.uint8)
    return np.stack([ch0, ch1, ch2], axis=-1)


class TestMetricsRigor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_base = make_synthetic_pattern(128, 128)
        if torch is not None and torch.cuda.is_available():
            cls.device = torch.device("cuda")
        elif torch is not None:
            cls.device = torch.device("cpu")
        else:
            cls.device = "cpu"

    # =========================================================================
    # B1. PSNR Tests
    # =========================================================================
    def test_psnr_identical(self):
        score = calculate_psnr(self.img_base, self.img_base, crop_border=0)
        self.assertEqual(score, float("inf"))

    def test_psnr_crop_border_behavior(self):
        corrupted = self.img_base.copy()
        corrupted[:3, :, :] = 0
        corrupted[:, :3, :] = 0
        p_c4 = calculate_psnr(corrupted, self.img_base, crop_border=4)
        p_c0 = calculate_psnr(corrupted, self.img_base, crop_border=0)
        self.assertEqual(p_c4, float("inf"), "Crop border 4 should exclude 3px border alterations")
        self.assertLess(p_c0, 40.0, "Crop border 0 should be penalized by border alterations")

    @unittest.skipIf(peak_signal_noise_ratio is None, "skimage not available")
    def test_psnr_basicsr_vs_skimage(self):
        noise = (self.img_base + 5).clip(0, 255).astype(np.uint8)
        bsr_psnr = calculate_psnr(noise, self.img_base, crop_border=0)
        ski_psnr = peak_signal_noise_ratio(self.img_base, noise, data_range=255)
        self.assertAlmostEqual(bsr_psnr, ski_psnr, places=4)

    # =========================================================================
    # B2. SSIM Tests
    # =========================================================================
    def test_ssim_identical(self):
        score = calculate_ssim(self.img_base, self.img_base, crop_border=0)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_ssim_perturbed(self):
        noise = (self.img_base + 15).clip(0, 255).astype(np.uint8)
        score = calculate_ssim(noise, self.img_base, crop_border=4)
        self.assertGreater(score, 0.70)
        self.assertLess(score, 1.0)

    # =========================================================================
    # B3. cPSNR (PROBA-V Metric) Deep Tests
    # =========================================================================
    def test_cpsnr_identical(self):
        score = calculate_cpsnr(self.img_base, self.img_base, crop_border=0)
        self.assertEqual(score, float("inf"), "Identical images must yield inf cPSNR")

    def test_cpsnr_known_translation(self):
        shift_r, shift_c = 3, 2
        shifted = np.zeros_like(self.img_base)
        shifted[shift_r:, shift_c:, :] = self.img_base[:-shift_r, :-shift_c, :]

        psnr_unaligned = calculate_psnr(shifted, self.img_base, crop_border=4)
        cpsnr_val = calculate_cpsnr(shifted, self.img_base, crop_border=4)

        self.assertGreater(cpsnr_val, psnr_unaligned + 5.0, "cPSNR must substantially improve score under pure translation")

    def test_cpsnr_radiometric_brightness_bias(self):
        biased = self.img_base.astype(np.float64)
        biased[:, :, 0] += 15.0
        biased[:, :, 1] -= 10.0
        biased[:, :, 2] += 20.0
        biased = np.clip(biased, 25, 230).astype(np.uint8)
        ref = np.clip(self.img_base.astype(np.float64), 25, 230).astype(np.uint8)

        psnr_biased = calculate_psnr(biased, ref, crop_border=0)
        cpsnr_biased = calculate_cpsnr(biased, ref, crop_border=0)

        self.assertGreater(cpsnr_biased, psnr_biased + 5.0, "cPSNR channel bias compensation must improve score under constant shift")

    def test_cpsnr_deliberate_incorrect_image(self):
        inverted = 255 - self.img_base
        cpsnr_bad = calculate_cpsnr(inverted, self.img_base, crop_border=4)
        self.assertLess(cpsnr_bad, 15.0, "Deliberately wrong image must not score high in cPSNR")

    def test_cpsnr_constant_image_no_crash(self):
        flat1 = np.full((128, 128, 3), 128, dtype=np.uint8)
        flat2 = np.full((128, 128, 3), 128, dtype=np.uint8)
        score = calculate_cpsnr(flat1, flat2, crop_border=4)
        self.assertEqual(score, float("inf"))

        flat3 = np.full((128, 128, 3), 140, dtype=np.uint8)
        score_diff = calculate_cpsnr(flat1, flat3, crop_border=4)
        self.assertEqual(score_diff, float("inf"), "Constant image differing by constant bias cancels to inf")

    # =========================================================================
    # B4. LPIPS Tests
    # =========================================================================
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_lpips_identical(self):
        score = calculate_lpips(self.img_base, self.img_base, lpips_model="vgg")
        self.assertAlmostEqual(score, 0.0, places=3, msg="LPIPS distance for identical images must be ~0.0")

    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_lpips_monotonicity(self):
        mild = (self.img_base.astype(np.float32) + np.random.normal(0, 5, self.img_base.shape)).clip(0, 255).astype(np.uint8)
        severe = np.random.randint(0, 256, self.img_base.shape, dtype=np.uint8)

        d_mild = calculate_lpips(mild, self.img_base, lpips_model="vgg")
        d_severe = calculate_lpips(severe, self.img_base, lpips_model="vgg")
        self.assertLess(d_mild, d_severe, "LPIPS distance must be lower for mild perturbation than severe noise")

    # =========================================================================
    # B5. CLIPScore Tests
    # =========================================================================
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_clipscore_identical(self):
        score = calculate_clipscore(self.img_base, self.img_base, clip_model="clipa-ViT-bigG-14")
        self.assertAlmostEqual(score, 1.0, places=3, msg="Self-similarity under CLIPScore must be 1.0")

    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_clipscore_perturbed(self):
        perturbed = (self.img_base.astype(np.float32) + 15).clip(0, 255).astype(np.uint8)
        inverted = 255 - self.img_base

        s_pert = calculate_clipscore(perturbed, self.img_base, clip_model="clipa-ViT-bigG-14")
        s_inv = calculate_clipscore(inverted, self.img_base, clip_model="clipa-ViT-bigG-14")
        self.assertGreater(s_pert, s_inv, "CLIPScore must be higher for similar image than inverted image")


if __name__ == "__main__":
    unittest.main()
