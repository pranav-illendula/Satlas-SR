"""
Satlas-SR Preprocessing and Ingestion Package.
Provides utilities for dataset health inspection, paired tile verification,
spatial registration audits, and raw Sentinel-2 raster reprojections.
"""
from __future__ import annotations

from .inspect_data import inspect_dataset
from .audit_dataset import audit_dataset_directory, audit_s2_tile
from .audit_spatial_registration import (
    compute_phase_correlation,
    compute_normalized_cross_correlation,
    audit_pair_registration,
)

__all__ = [
    "inspect_dataset",
    "audit_dataset_directory",
    "audit_s2_tile",
    "compute_phase_correlation",
    "compute_normalized_cross_correlation",
    "audit_pair_registration",
]

