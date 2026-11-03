#!/usr/bin/env python3
"""
Statistical Analysis Tool for Satlas-SR Benchmarks.
Computes 95% non-parametric bootstrap confidence intervals, median, interquartile ranges,
and identifies failure outliers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(data: np.ndarray, num_resamples: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Computes non-parametric percentile bootstrap confidence intervals."""
    rng = np.random.default_rng(42)
    n = len(data)
    boot_indices = rng.integers(0, n, size=(num_resamples, n))
    boot_means = np.mean(data[boot_indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return low, high


def analyze_metrics_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    results = {}
    for col in rows[0].keys():
        try:
            values = np.array([float(r[col]) for r in rows if r[col] not in (None, "", "None")])
            if len(values) > 0:
                low_ci, high_ci = bootstrap_ci(values)
                results[col] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "bootstrap_95_ci": [low_ci, high_ci],
                }
        except ValueError:
            continue

    print(f"\n--- Statistical Distribution Report ({csv_path.name}) ---")
    for metric, stats in results.items():
        ci_str = f"[{stats['bootstrap_95_ci'][0]:.3f}, {stats['bootstrap_95_ci'][1]:.3f}]"
        print(f"{metric:25s}: Mean {stats['mean']:8.3f} ± {stats['std']:6.3f} | Median {stats['median']:8.3f} | 95% CI {ci_str}")

    return results


def main() -> int:
    default_csv = Path("results/satlas_sr_revalidation/per_tile_metrics.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=default_csv, help=f"Path to per-tile metrics CSV (default: {default_csv})")
    parser.add_argument("--output-json", type=Path, default=None, help="Save distribution stats to JSON")
    args = parser.parse_args()

    stats = analyze_metrics_csv(args.csv_path)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

