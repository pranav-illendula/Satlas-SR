# Satlas-SR Windows Compatibility & Path-Handling Audit Report

- **Date**: 2026-10-22
- **Audit Target**: `ssr/data/s2-naip_dataset.py`, `ssr/infer.py`, and Windows filesystem operations
- **Test Suite**: `tests/test_dataset_path_compatibility.py` (3 unit tests, all passing)

---

## 1. Context & Identified Defects

In the official Satlas-SR codebase, file paths were processed using hard-coded forward slashes (`/`), assuming a POSIX execution environment (e.g., Linux or macOS containers).

When deployed on Windows NT systems:
1. `glob.glob(...)` returns path strings with Windows backslashes (e.g., `C:\data\naip\20771_46525\rgb.png`).
2. Performing `.split('/')` on Windows path strings fails to partition directory components:
   - `n.split('/')` yields a list of length 1: `['C:\\data\\naip\\20771_46525\\rgb.png']`.
   - In `S2NAIPDataset.__init__`, attempting to access `split_path[-2]` immediately throws `IndexError: list index out of range`, completely crashing dataset initialization on Windows.
3. Similar string operations occurred in `old_chip = old_naip.split('/')[-1][:-4]` and `get_tile_weight_sampler`.

---

## 2. Minimal, Backward-Compatible Fix

To preserve full Linux/macOS compatibility while providing native Windows operation, fragile string splits were replaced with standard Python `os.path` functions:

```python
# Before (POSIX assumption, fails on Windows):
split_path = n.split('/')
chip = split_path[-2]

# After (Cross-platform, verified on Windows & Linux):
chip = os.path.basename(os.path.dirname(n))
```

And for chip stems:
```python
# Before:
old_chip = old_naip.split('/')[-1][:-4]

# After:
old_chip = os.path.splitext(os.path.basename(old_naip))[0]
```

### Verification Matrix

| Path Input Format | Example Path String | Legacy `.split('/')` Behavior | Robust `os.path` Behavior | Status |
| :--- | :--- | :--- | :--- | :--- |
| **POSIX Style** | `/data/small_val_set/naip/20771_46525/rgb.png` | `20771_46525` | `20771_46525` | **PASS** |
| **Windows Style** | `C:\data\small_val_set\naip\20771_46525\rgb.png` | `IndexError: out of range` | `20771_46525` | **PASS** |
| **Mixed Separator** | `C:/data\small_val_set/naip\38649_48273/rgb.png` | `Unpredictable split` | `38649_48273` | **PASS** |

---

## 3. Scope of Impact & Invariants Preserved

- **Zero Breaking Changes**: The change is 100% backward-compatible with Linux training jobs.
- **Inference Pipeline Unaffected**: The inference pipeline (`run_full_validation_inference.py`) uses `pathlib.Path` and was previously engineered to avoid these defects.
- **No Heavy Pipeline Re-execution**: No weights, checkpoints, or inferences were rerun or modified.
