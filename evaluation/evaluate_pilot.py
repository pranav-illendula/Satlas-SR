#!/usr/bin/env python3
"""
Lightweight evaluation runner for rapid smoke testing on pilot tiles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluation.evaluate_validation import evaluate_predictions_directory

PILOT_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]


def main() -> int:
    default_pred = repo_root / "satlas-super-resolution" / "results" / "satlas_sr_pilot_five_tiles"
    default_data = repo_root / "data" / "small_val_set"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=default_pred, help=f"Predictions path (default: {default_pred})")
    parser.add_argument("--dataset", type=Path, default=default_data, help=f"Dataset path containing naip/ (default: {default_data})")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV")
    args = parser.parse_args()

    evaluate_predictions_directory(args.predictions, args.dataset, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
