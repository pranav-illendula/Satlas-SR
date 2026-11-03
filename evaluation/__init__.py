"""
Evaluation Suite for Satlas Super-Resolution.
Provides validation evaluators, official paper metrics (cPSNR, LPIPS, CLIPScore),
and non-parametric statistical analyzers.
"""
from __future__ import annotations

from .metrics.cpsnr import calculate_cpsnr
from .metrics.lpips import calculate_lpips
from .metrics.clipscore import calculate_clipscore
from .evaluate_validation import evaluate_predictions_directory
from .evaluate_paper_metrics import evaluate_paper_metrics
from .run_statistical_analysis import analyze_metrics_csv, bootstrap_ci

__all__ = [
    "calculate_cpsnr",
    "calculate_lpips",
    "calculate_clipscore",
    "evaluate_predictions_directory",
    "evaluate_paper_metrics",
    "analyze_metrics_csv",
    "bootstrap_ci",
]

