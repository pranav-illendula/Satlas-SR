# Research Reports & Scientific Provenance Audits

This directory contains research reports, provenance investigations, gap analyses, and reproduction runbooks tracking the experimental trajectory from baseline reproduction to uncertainty-guided reconstruction.

---

## 🌟 Master Consolidated Research Report

All 18 historical research reports and technical audits have been unified, synthesized, and merged into a single comprehensive master document:

👉 [**`MASTER_CONSOLIDATED_RESEARCH_REPORT.md`**](MASTER_CONSOLIDATED_RESEARCH_REPORT.md)

This authoritative document covers:
1. **Executive Synthesis & Report Concordance**: Dashboard and mapping across all historical studies.
2. **Technical Architecture Deep Dive**: Sensor physics, Web-Mercator coordinate tiling, 23-block RRDBNet generator, Spectral Norm U-Net discriminator, and RaGAN loss dynamics.
3. **Dataset Provenance & Lineage**: Origin of `super_resolution_small_val_set_2024-01-09.tar`, unanchored metadata audits, and the spatial registration resolution mismatch correction.
4. **Reproduction Feasibility & Baseline Audit**: SHA-256 weight verification (`esrgan_8S2.pth`), 5-tile pilot verification, and preprocessing fidelity.
5. **Full Validation Benchmark (256 Tiles)**: Paper reproduction metrics (cPSNR: $20.21 \pm 4.14\\,\\text{dB}$, LPIPS: $0.347$, CLIPScore: $0.961$) and paper gap analysis.
6. **Metric Deep Dive & Mathematical Proofs**: Proof of cPSNR translation and brightness bias invariance, border crop semantics, and CLIPScore bounds.
7. **Statistical Distribution Analysis**: Non-parametric bootstrap resampling ($N = 256$), skewness, and pairwise correlation matrices.
8. **Per-Tile Failure Mode & Disagreement Analysis**: Extreme tile rankings, moving object hallucinations, and cloud shadow artifacts.
9. **Platform Compatibility & Infrastructure Hardening**: Cross-platform path handling and headless pure-NumPy fallbacks.
10. **Master Reproduction Runbook**: Step-by-step CLI execution guide.

---

## Directory Organization

```
reports/
├── README.md                               # This documentation index
├── MASTER_CONSOLIDATED_RESEARCH_REPORT.md  # Single consolidated master research report
├── five_tile_visual_audit.png              # Visual comparison crops from the pilot audit
├── reconstructed_tci_frame_indices.csv     # Reconstructed frame indices for validation tiles
└── individual_reports/                     # Archive of individual historical report files
    ├── TECHNICAL_APPROACH_DEEP_DIVE.md
    ├── master_reproduction_audit.md
    ├── satlas_sr_base_paper_reproduction_gap.md
    ├── SATLAS_SR_COMPREHENSIVE_REPRODUCTION_REPORT_v2.md
    ├── satlas_sr_full_validation_evaluation.md
    ├── satlas_sr_metric_validation_deep_dive.md
    ├── satlas_sr_model_fidelity_audit.md
    ├── satlas_sr_paper_comparability.md
    ├── satlas_sr_per_tile_failure_analysis.md
    ├── satlas_sr_provenance_investigation.md
    ├── satlas_sr_provenance_request.md
    ├── satlas_sr_reproduction_feasibility.md
    ├── satlas_sr_reproduction_runbook.md
    ├── satlas_sr_spatial_registration_reassessment.md
    ├── satlas_sr_statistical_analysis.md
    ├── satlas_sr_windows_compatibility_audit.md
    ├── five_tile_evaluation_audit.md
    └── maintainer_provenance_request.md
```

Individual archived files are preserved in [`individual_reports/`](individual_reports/) for historical record and granular section reference.
