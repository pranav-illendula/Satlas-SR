"""
Rigorous synthetic and numerical test suite for Satlas-SR metrics:
- PSNR (BasicSR vs skimage)
- SSIM (BasicSR vs skimage)
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
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Local imports from repo
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from ssr.metrics.cpsnr import calculate_cpsnr
from ssr.metrics.lpips import calculate_lpips
from ssr.metrics.clipscore import calculate_clipscore


def make_synthetic_pattern(h: int = 128, w: int = 128) -> np.ndarray:
    """Generate a rich, deterministic synthetic texture with edges, gradients, and patterns."""
    y, x = np.mgrid[:h, :w]
    # Combination of sinusoids, gradients, and sharp steps
    ch0 = (np.sin(x / 4.0) * np.cos(y / 4.0) * 100 + 128).astype(np.uint8)
    ch1 = ((x + y) * 255 / (h + w)).astype(np.uint8)
    ch2 = (((x // 8) % 2) ^ ((y // 8) % 2)) * 200 + 30
    ch2 = ch2.astype(np.uint8)
    return np.stack([ch0, ch1, ch2], axis=-1)


class TestMetricsRigor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_base = make_synthetic_pattern(128, 128)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning Metric Rigor Tests on device: {cls.device}")

    # =========================================================================
    # B1. PSNR Tests
    # =========================================================================
    def test_psnr_identical(self):
        score = calculate_psnr(self.img_base, self.img_base, crop_border=0)
        self.assertEqual(score, float("inf"))

    def test_psnr_crop_border_behavior(self):
        # Alter only the border (first 3 pixels)
        corrupted = self.img_base.copy()
        corrupted[:3, :, :] = 0
        corrupted[:, :3, :] = 0
        # With crop_border=4, the corrupted border is excluded -> perfect score
        p_c4 = calculate_psnr(corrupted, self.img_base, crop_border=4)
        p_c0 = calculate_psnr(corrupted, self.img_base, crop_border=0)
        self.assertEqual(p_c4, float("inf"), "Crop border 4 should exclude 3px border alterations")
        self.assertLess(p_c0, 40.0, "Crop border 0 should be penalized by border alterations")

    def test_psnr_basicsr_vs_skimage(self):
        # Verify BasicSR PSNR (crop_border=0) matches skimage peak_signal_noise_ratio
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
        # Shift image by (+3, +2) pixels within allowed max_offset=8
        shift_r, shift_c = 3, 2
        shifted = np.zeros_like(self.img_base)
        shifted[shift_r:, shift_c:, :] = self.img_base[:-shift_r, :-shift_c, :]

        psnr_unaligned = calculate_psnr(shifted, self.img_base, crop_border=4)
        cpsnr_val = calculate_cpsnr(shifted, self.img_base, crop_border=4)

        print(f"\n[cPSNR Shift Test] Shift=(+{shift_r}, +{shift_c}): Unaligned PSNR={psnr_unaligned:.2f} dB, cPSNR={cpsnr_val:.2f} dB")
        self.assertGreater(cpsnr_val, psnr_unaligned + 5.0, "cPSNR must substantially improve score under pure translation")

    def test_cpsnr_radiometric_brightness_bias(self):
        # Shift each channel by a constant bias: R+15, G-10, B+20
        biased = self.img_base.astype(np.float64)
        biased[:, :, 0] += 15.0
        biased[:, :, 1] -= 10.0
        biased[:, :, 2] += 20.0
        biased = np.clip(biased, 25, 230).astype(np.uint8)
        ref = np.clip(self.img_base.astype(np.float64), 25, 230).astype(np.uint8)

        psnr_biased = calculate_psnr(biased, ref, crop_border=0)
        cpsnr_biased = calculate_cpsnr(biased, ref, crop_border=0)

        print(f"\n[cPSNR Bias Test] Biased PSNR={psnr_biased:.2f} dB, cPSNR={cpsnr_biased:.2f} dB")
        self.assertGreater(cpsnr_biased, psnr_biased + 5.0, "cPSNR channel bias compensation must improve score under constant shift")

    def test_cpsnr_deliberate_incorrect_image(self):
        # Inverted image
        inverted = 255 - self.img_base
        cpsnr_bad = calculate_cpsnr(inverted, self.img_base, crop_border=4)
        print(f"\n[cPSNR Inverted Test] cPSNR={cpsnr_bad:.2f} dB")
        self.assertLess(cpsnr_bad, 15.0, "Deliberately wrong image must not score high in cPSNR")

    def test_cpsnr_constant_image_no_crash(self):
        # Flat constant image should not crash with NaN or ZeroDivisionError
        flat1 = np.full((128, 128, 3), 128, dtype=np.uint8)
        flat2 = np.full((128, 128, 3), 128, dtype=np.uint8)
        score = calculate_cpsnr(flat1, flat2, crop_border=4)
        self.assertEqual(score, float("inf"))

        flat3 = np.full((128, 128, 3), 140, dtype=np.uint8)
        score_diff = calculate_cpsnr(flat1, flat3, crop_border=4)
        # Because bias is cancelled, flat images differing by constant offset yield identical content
        self.assertEqual(score_diff, float("inf"), "Constant image differing by constant bias cancels to inf")

    # =========================================================================
    # B4. LPIPS Tests
    # =========================================================================
    def test_lpips_identical(self):
        score = calculate_lpips(self.img_base, self.img_base, lpips_model="vgg")
        print(f"\n[LPIPS Identical] distance={score:.6f}")
        self.assertAlmostEqual(score, 0.0, places=3, msg="LPIPS distance for identical images must be ~0.0")

    def test_lpips_monotonicity(self):
        mild = (self.img_base.astype(np.float32) + np.random.normal(0, 5, self.img_base.shape)).clip(0, 255).astype(np.uint8)
        severe = np.random.randint(0, 256, self.img_base.shape, dtype=np.uint8)

        d_mild = calculate_lpips(mild, self.img_base, lpips_model="vgg")
        d_severe = calculate_lpips(severe, self.img_base, lpips_model="vgg")
        print(f"\n[LPIPS Monotonicity] d_mild={d_mild:.4f}, d_severe={d_severe:.4f}")
        self.assertLess(d_mild, d_severe, "LPIPS distance must be lower for mild perturbation than severe noise")

    # =========================================================================
    # B5. CLIPScore Tests
    # =========================================================================
    def test_clipscore_identical(self):
        score = calculate_clipscore(self.img_base, self.img_base, clip_model="clipa-ViT-bigG-14")
        print(f"\n[CLIPScore Identical] cosine_sim={score:.6f}")
        self.assertAlmostEqual(score, 1.0, places=3, msg="Self-similarity under CLIPScore must be 1.0")

    def test_clipscore_perturbed(self):
        perturbed = (self.img_base.astype(np.float32) + 15).clip(0, 255).astype(np.uint8)
        inverted = 255 - self.img_base

        s_pert = calculate_clipscore(perturbed, self.img_base, clip_model="clipa-ViT-bigG-14")
        s_inv = calculate_clipscore(inverted, self.img_base, clip_model="clipa-ViT-bigG-14")

        print(f"\n[CLIPScore Discrimination] s_perturbed={s_pert:.4f}, s_inverted={s_inv:.4f}")
        self.assertGreater(s_pert, s_inv, "CLIPScore must be higher for similar image than inverted image")


if __name__ == "__main__":
    unittest.main()
