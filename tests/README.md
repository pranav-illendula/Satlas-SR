# Unit & Integration Tests

This directory contains automated unit tests verifying dataset path compatibility, data loader resilience, and mathematical invariants of metric implementations.

## Test Suites

| Test File | Target Module | Scope & Invariants Tested |
| :--- | :--- | :--- |
| `test_dataset_path_compatibility.py` | `ssr.data` / `evaluation` | Verifies data loader resolution across diverse directory layouts, checking path joining, fallback handling, and tile-pairing logic. |
| `test_metrics_rigor.py` | `evaluation.metrics` / `ssr.metrics` | Validates fundamental mathematical invariants of cPSNR and PSNR: identity mapping ($cPSNR = \infty$), translation invariance ($\pm 8$ px), brightness bias compensation, and boundary crop semantics. |

## Running Tests

Execute all tests from the repository root:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run an individual test suite:
```bash
python -m unittest tests/test_dataset_path_compatibility.py
python -m unittest tests/test_metrics_rigor.py
```

