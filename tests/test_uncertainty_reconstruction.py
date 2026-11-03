"""
Unit tests for the Uncertainty Calibration & Guided Reconstruction package:
- D4 spatial transformation invertibility (8 group elements)
- Spatial uncertainty computation
- Temporal revisit subsampling & uncertainty computation
- Calibrated hybrid uncertainty map synthesis
- Brent's method budget conservation for soft blending
- Boundary seam step discontinuity metrics
- Registration shift recovery
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np

from technical_architecture.uncertainty_and_reconstruction import (
    apply_d4_forward_numpy,
    apply_d4_inverse_numpy,
    compute_spatial_uncertainty_from_tta,
    generate_temporal_subsamples,
    compute_temporal_uncertainty,
    compute_hybrid_uncertainty,
    HybridUncertaintyCalibrator,
    compute_sobel_gradient,
    stable_sigmoid,
    solve_tau_for_budget,
    apply_soft_blending,
    apply_hard_thresholding,
    compute_boundary_step_discontinuity,
    compute_registration_shift,
    evaluate_reconstruction_quality,
)


class TestUncertaintyReconstruction(unittest.TestCase):
    def setUp(self):
        # Create deterministic synthetic pattern of shape (64, 64, 3)
        y, x = np.mgrid[:64, :64]
        ch0 = (np.sin(x / 4.0) * np.cos(y / 4.0) * 100 + 128).astype(np.uint8)
        ch1 = ((x + y) * 255 / 128).astype(np.uint8)
        ch2 = (((x // 4) % 2) ^ ((y // 4) % 2)) * 200 + 30
        ch2 = ch2.astype(np.uint8)
        self.img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32) / 255.0

    # =========================================================================
    # 1. D4 Transformation Invertibility Tests
    # =========================================================================
    def test_d4_all_eight_inverses(self):
        """Verify that apply_d4_inverse(apply_d4_forward(x)) == x for all 8 D4 elements."""
        for k in [0, 1, 2, 3]:
            for f in [0, 1]:
                fwd = apply_d4_forward_numpy(self.img, k=k, f=f)
                inv = apply_d4_inverse_numpy(fwd, k=k, f=f)
                np.testing.assert_allclose(
                    inv, self.img, atol=1e-6,
                    err_msg=f"Failed invertibility for D4 element (k={k}, f={f})"
                )

    # =========================================================================
    # 2. Spatial Uncertainty Tests
    # =========================================================================
    def test_spatial_uncertainty_identical(self):
        """Identical predictions across 8 orientations must yield 0.0 uncertainty."""
        preds = [self.img.copy() for _ in range(8)]
        u_spatial, mean_pred = compute_spatial_uncertainty_from_tta(preds)
        self.assertEqual(u_spatial.shape, (64, 64))
        np.testing.assert_allclose(u_spatial, 0.0, atol=1e-5)
        np.testing.assert_allclose(mean_pred, self.img, atol=1e-7)

    def test_spatial_uncertainty_perturbed(self):
        """Perturbed predictions must produce positive uncertainty map."""
        preds = []
        for i in range(8):
            noise = np.full_like(self.img, fill_value=i * 0.05)
            preds.append((self.img + noise).clip(0.0, 1.0))
        u_spatial, mean_pred = compute_spatial_uncertainty_from_tta(preds)
        self.assertTrue(np.all(u_spatial > 0.0))

    # =========================================================================
    # 3. Temporal Uncertainty Tests
    # =========================================================================
    def test_generate_temporal_subsamples(self):
        """Verify deterministic subsampling generation."""
        subsamples = generate_temporal_subsamples(list(range(32)), num_subsamples=6, frames_per_sample=8, seed=42)
        self.assertEqual(len(subsamples), 7)  # S0 + 6 alternative
        for s in subsamples:
            self.assertEqual(len(s), 8)

    def test_temporal_uncertainty_calculation(self):
        """Verify temporal variance and ensemble mean calculation."""
        preds = [self.img + 0.02 * i for i in range(7)]
        u_temp, ens_mean = compute_temporal_uncertainty(preds)
        self.assertEqual(u_temp.shape, (64, 64))
        self.assertEqual(ens_mean.shape, (64, 64, 3))
        self.assertTrue(np.all(u_temp > 0.0))

    # =========================================================================
    # 4. Calibrated Hybrid Uncertainty Tests
    # =========================================================================
    def test_hybrid_uncertainty_fusion(self):
        """Verify standardized hybrid map calculation and alpha weighting."""
        u_spat = np.random.uniform(0.01, 0.15, size=(64, 64)).astype(np.float32)
        u_temp = np.random.uniform(0.01, 0.10, size=(64, 64)).astype(np.float32)
        u_hyb, z_s, z_t = compute_hybrid_uncertainty(u_spat, u_temp, alpha=0.19)

        self.assertEqual(u_hyb.shape, (64, 64))
        expected = 0.19 * z_s + 0.81 * z_t
        np.testing.assert_allclose(u_hyb, expected, atol=1e-5)

    def test_hybrid_calibrator_fit_and_predict(self):
        """Verify HybridUncertaintyCalibrator normalization fitting."""
        calib = HybridUncertaintyCalibrator(alpha=0.19)
        spat_maps = [np.random.normal(0.08, 0.05, (64, 64)) for _ in range(5)]
        temp_maps = [np.random.normal(0.05, 0.04, (64, 64)) for _ in range(5)]
        stats_dict = calib.fit_normalization(spat_maps, temp_maps)

        self.assertAlmostEqual(stats_dict["mu_spatial"], 0.08, delta=0.03)
        self.assertAlmostEqual(stats_dict["mu_temporal"], 0.05, delta=0.03)

        pred_map = calib.predict(spat_maps[0], temp_maps[0])
        self.assertEqual(pred_map.shape, (64, 64))

    # =========================================================================
    # 5. Soft-Blending Reconstruction & Budget Conservation Tests
    # =========================================================================
    def test_solve_tau_budget_conservation(self):
        """Verify that Brent's method satisfies the target intervention budget exactly."""
        u_map = np.random.normal(0.0, 1.0, size=(64, 64)).astype(np.float32)
        beta = 0.20
        target_budget = 0.20

        tau_star = solve_tau_for_budget(u_map.flatten(), beta=beta, target_budget=target_budget)
        actual_weights = stable_sigmoid((u_map - tau_star) / beta)
        actual_budget = float(np.mean(actual_weights))

        self.assertAlmostEqual(actual_budget, target_budget, places=4,
                               msg="Budget conservation violated!")

    def test_apply_soft_blending_outputs(self):
        """Verify soft-blending reconstruction dimensions, value bounds, and linearity."""
        s0 = self.img.copy()
        ens = (self.img * 0.9 + 0.05).clip(0.0, 1.0)
        u_map = np.random.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)

        recon, weights, tau = apply_soft_blending(s0, ens, u_map, budget=0.20, beta=0.20)

        self.assertEqual(recon.shape, (64, 64, 3))
        self.assertEqual(weights.shape, (64, 64))
        self.assertTrue(np.all(weights >= 0.0) and np.all(weights <= 1.0))
        self.assertTrue(np.all(recon >= 0.0) and np.all(recon <= 1.0))

    def test_apply_hard_thresholding(self):
        """Verify discrete top-k thresholding replaces exactly the target budget fraction."""
        s0 = self.img.copy()
        ens = (self.img * 0.9 + 0.05).clip(0.0, 1.0)
        u_map = np.random.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)

        recon, mask, cutoff = apply_hard_thresholding(s0, ens, u_map, budget=0.25)
        replaced_frac = float(np.mean(mask))
        self.assertAlmostEqual(replaced_frac, 0.25, delta=0.01)

    # =========================================================================
    # 6. Boundary Discontinuity & Quality Metrics Tests
    # =========================================================================
    def test_sobel_gradient_calculation(self):
        """Verify 2D Sobel gradient computation on constant vs stepped image."""
        flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
        grad_flat = compute_sobel_gradient(flat)
        np.testing.assert_allclose(grad_flat, 0.0, atol=1e-6)

        stepped = flat.copy()
        stepped[:, 16:, :] = 1.0
        grad_step = compute_sobel_gradient(stepped)
        self.assertTrue(np.max(grad_step) > 0.0)

    def test_boundary_step_discontinuity(self):
        """Verify that a sharp step mask creates a measurable seam discontinuity."""
        flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[8:24, 8:24] = 1.0

        # With an image having an abrupt jump on the mask boundary
        corrupted = flat.copy()
        corrupted[mask > 0.5] = 1.0

        seam_score = compute_boundary_step_discontinuity(corrupted, mask)
        self.assertGreater(seam_score, 0.0)

    def test_registration_shift_recovery(self):
        """Verify that integer translation shift minimization recovers known shifts."""
        ref = self.img.copy()
        pred = np.zeros_like(ref)
        shift_y, shift_x = 2, 1
        # Set pred interior to match ref offset by (shift_y, shift_x)
        pred[4:60, 4:60] = ref[4+shift_y:60+shift_y, 4+shift_x:60+shift_x]

        dx, dy, min_mse = compute_registration_shift(pred, ref, max_shift=4, border=4)
        self.assertEqual(dx, shift_x)
        self.assertEqual(dy, shift_y)
        self.assertAlmostEqual(min_mse, 0.0, places=6)

    def test_evaluate_reconstruction_quality(self):
        """Verify metric evaluation dictionary outputs."""
        metrics = evaluate_reconstruction_quality(self.img, self.img, border=4, crop_size=56)
        self.assertEqual(metrics["psnr_db"], 99.0)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=7)
        self.assertGreater(metrics["edge_sharpness"], 0.0)


if __name__ == "__main__":
    unittest.main()
