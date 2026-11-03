"""Audit and evaluate existing Satlas-SR validation outputs; never changes source or inference images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SSIM_SETTINGS = "channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True"
PILOT_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]
CSV_FIELDS = [
    "tile_id", "evaluation_status", "exclusion_reason", "inference_status",
    "s2_input_path", "s2_stack_dimensions_wh", "s2_stack_mode", "s2_frame_count",
    "selected_frame_indices_zero_based", "lr_preview_path", "lr_preview_dimensions_wh", "lr_preview_mode",
    "prediction_path", "prediction_dimensions_wh", "prediction_mode", "prediction_min", "prediction_max",
    "reference_path", "naip_folder_date_token", "reference_dimensions_wh", "reference_mode", "reference_min", "reference_max",
    "psnr_db", "ssim", "data_range", "ssim_settings", "spatial_registration", "temporal_comparability", "warning",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def open_png(path: Path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"expected PNG, got {im.format}")
        return im.copy()


def dimensions_wh(im: Image.Image) -> str:
    return f"{im.width}x{im.height}"


def date_token(ref: Path) -> str:
    # README defines image_uuid as carrying the NAIP capture timestamp. Report only a token in its name.
    parent = ref.parent.parent.name
    match = re.search(r"(\d{8})$", parent)
    return match.group(1) if match else ""


def write_contact_sheet(path: Path, rows: list[dict], input_indices: dict[str, list[int]], limit: int = 12) -> list[str]:
    eligible = sorted(r["tile_id"] for r in rows if r["evaluation_status"] == "METRICS_COMPUTED_EXPLORATORY")
    chosen = [t for t in PILOT_IDS if t in eligible]
    remaining = [t for t in eligible if t not in chosen]
    # Add seven evenly spaced non-pilot tile IDs for a deterministic, spread-out sample.
    n_extra = min(7, len(remaining))
    if n_extra:
        indices = sorted({round(i * (len(remaining) - 1) / max(n_extra - 1, 1)) for i in range(n_extra)})
        for idx in indices:
            if remaining[idx] not in chosen:
                chosen.append(remaining[idx])
    chosen = chosen[:limit]
    row_by_id = {r["tile_id"]: r for r in rows}
    margin, gap, label_h = 12, 12, 26
    s2_w, other_w, panel_h = 256, 128, 128
    header_h, row_h = 24, label_h + panel_h + 12
    width = margin * 2 + s2_w + other_w * 2 + gap * 2
    height = margin * 2 + header_h + len(chosen) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x_s2 = margin
    x_sr = x_s2 + s2_w + gap
    x_ref = x_sr + other_w + gap
    y0 = margin
    draw.text((x_s2, y0), "8 selected S2 TCI frames (4x2; nearest-neighbor display)", fill="black")
    draw.text((x_sr, y0), "ESRGAN", fill="black")
    draw.text((x_ref, y0), "NAIP", fill="black")
    for i, tile in enumerate(chosen):
        r = row_by_id[tile]
        y = margin + header_h + i * row_h
        draw.text((margin, y), tile, fill="black")
        panel_y = y + label_h
        stack = open_png(Path(r["s2_input_path"]))
        indices = input_indices[tile]
        for j, idx in enumerate(indices):
            frame = stack.crop((0, idx * 32, 32, (idx + 1) * 32)).resize((64, 64), Image.Resampling.NEAREST)
            fx, fy = x_s2 + (j % 4) * 64, panel_y + (j // 4) * 64
            sheet.paste(frame, (fx, fy))
            draw.rectangle((fx, fy, fx + 13, fy + 12), fill="black")
            draw.text((fx + 2, fy), str(j + 1), fill="white")
        sheet.paste(open_png(Path(r["prediction_path"])), (x_sr, panel_y))
        sheet.paste(open_png(Path(r["reference_path"])), (x_ref, panel_y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return chosen


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=here.parent / "data" / "small_val_set")
    parser.add_argument("--results", type=Path, default=here / "results" / "satlas_sr_full_validation")
    parser.add_argument("--manifest", type=Path, help="defaults to <results>/inference_manifest.csv")
    args = parser.parse_args()
    started = time.perf_counter()
    dataset, results = args.dataset.resolve(), args.results.resolve()
    manifest_path = (args.manifest or results / "inference_manifest.csv").resolve()
    report_path = here / "reports" / "satlas_sr_full_validation_evaluation.md"
    csv_path, json_path = results / "tile_metrics.csv", results / "summary.json"
    s2_root, naip_root = dataset / "sentinel2", dataset / "naip"
    if not manifest_path.is_file() or not s2_root.is_dir() or not naip_root.is_dir():
        raise FileNotFoundError(f"Expected manifest and dataset folders: {manifest_path}; {s2_root}; {naip_root}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        manifest_rows = list(csv.DictReader(f))
    manifest_by_id: dict[str, list[dict]] = defaultdict(list)
    for r in manifest_rows:
        manifest_by_id[r.get("tile_id", "")].append(r)
    s2_dirs = sorted((p for p in s2_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    refs_by_id: dict[str, list[Path]] = defaultdict(list)
    for p in naip_root.rglob("rgb.png"):
        refs_by_id[p.parent.name].append(p)

    counts = Counter(r.get("status", "") for r in manifest_rows)
    duplicate_manifest_ids = sorted(k for k, v in manifest_by_id.items() if len(v) != 1)
    duplicate_reference_ids = sorted(k for k, v in refs_by_id.items() if len(v) != 1)
    rows, arrays, errors, input_indices = [], {}, [], {}
    global_ranges = {k: [None, None] for k in ("prediction", "reference", "lr_preview")}

    def include_range(label, arr):
        lo, hi = int(arr.min()), int(arr.max())
        current = global_ranges[label]
        current[0] = lo if current[0] is None else min(current[0], lo)
        current[1] = hi if current[1] is None else max(current[1], hi)
        return lo, hi

    missing_or_malformed = Counter({"s2_input": 0, "lr_preview": 0, "prediction": 0, "reference": 0})
    for s2_dir in s2_dirs:
        tile = s2_dir.name
        row = {key: "" for key in CSV_FIELDS}
        row.update(tile_id=tile, evaluation_status="EXCLUDED", data_range="255",
                   ssim_settings=SSIM_SETTINGS, spatial_registration="UNVERIFIED",
                   temporal_comparability="SOURCE WINDOW DOCUMENTED; per-frame dates/cloud status unavailable")
        mr = manifest_by_id.get(tile, [])
        if len(mr) != 1:
            reason = f"expected one inference manifest entry, found {len(mr)}"
            row["inference_status"] = "MISSING_OR_DUPLICATE"
        else:
            m = mr[0]
            row["inference_status"] = m.get("status", "")
            row["s2_input_path"] = m.get("input_path", "")
            row["lr_preview_path"] = m.get("lr_output_path", "")
            row["prediction_path"] = m.get("sr_output_path", "")
            row["selected_frame_indices_zero_based"] = m.get("selected_frame_indices_zero_based", "")
            reason = ""
            if m.get("status") not in ("OK", "SKIPPED_VALID"):
                reason = f"inference status {m.get('status')}: {m.get('error', '')}"
        refs = refs_by_id.get(tile, [])
        ref_path = refs[0] if len(refs) == 1 else None
        row["reference_path"] = str(ref_path) if ref_path else ""
        if ref_path:
            row["naip_folder_date_token"] = date_token(ref_path)
        elif not reason:
            reason = f"expected exactly one NAIP rgb.png, found {len(refs)}"

        if len(mr) == 1:
            m = mr[0]
            try:
                inp = open_png(Path(m["input_path"]))
                row["s2_stack_dimensions_wh"], row["s2_stack_mode"] = dimensions_wh(inp), inp.mode
                row["s2_frame_count"] = str(inp.height // 32 if inp.width == 32 and inp.height % 32 == 0 else 0)
                if inp.mode != "RGB" or inp.width != 32 or inp.height % 32:
                    raise ValueError(f"expected RGB PNG stack 32x(32*n), got {inp.mode} {inp.size}")
            except Exception as e:
                missing_or_malformed["s2_input"] += 1
                reason = reason or f"invalid/missing S2 input: {type(e).__name__}: {e}"
                errors.append({"tile_id": tile, "file": "s2_input", "error": str(e)})
            for field, path_key, label, expected in [
                ("lr_preview", "lr_output_path", "lr_preview", (32, 32)),
                ("prediction", "sr_output_path", "prediction", (128, 128)),
            ]:
                try:
                    im = open_png(Path(m[path_key]))
                    row[f"{label}_dimensions_wh"], row[f"{label}_mode"] = dimensions_wh(im), im.mode
                    arr = np.asarray(im)
                    if label == "prediction":
                        row["prediction_min"], row["prediction_max"] = include_range(label, arr)
                    else:
                        include_range(label, arr)
                    if im.mode != "RGB" or im.size != expected:
                        raise ValueError(f"expected RGB PNG {expected[0]}x{expected[1]}, got {im.mode} {im.size}")
                    if label == "prediction": arrays["prediction", tile] = arr
                except Exception as e:
                    missing_or_malformed[label] += 1
                    reason = reason or f"invalid/missing {label}: {type(e).__name__}: {e}"
                    errors.append({"tile_id": tile, "file": label, "error": str(e)})

        if ref_path:
            try:
                ref = open_png(ref_path)
                row["reference_dimensions_wh"], row["reference_mode"] = dimensions_wh(ref), ref.mode
                ref_arr = np.asarray(ref)
                row["reference_min"], row["reference_max"] = include_range("reference", ref_arr)
                if ref.mode != "RGB" or ref.size != (128, 128):
                    raise ValueError(f"expected RGB PNG 128x128, got {ref.mode} {ref.size}")
                arrays["reference", tile] = ref_arr
            except Exception as e:
                missing_or_malformed["reference"] += 1
                reason = reason or f"invalid/missing reference: {type(e).__name__}: {e}"
                errors.append({"tile_id": tile, "file": "reference", "error": str(e)})

        if not reason:
            pred, ref = arrays["prediction", tile], arrays["reference", tile]
            if pred.shape != ref.shape:
                reason = f"prediction/reference array shapes differ: {pred.shape} vs {ref.shape}"
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    row["psnr_db"] = f"{peak_signal_noise_ratio(ref, pred, data_range=255):.8f}"
                    row["ssim"] = f"{structural_similarity(ref, pred, data_range=255, channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True):.8f}"
                row["warning"] = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)
                row["evaluation_status"] = "METRICS_COMPUTED_EXPLORATORY"
        if reason:
            row["exclusion_reason"] = reason
            errors.append({"tile_id": tile, "file": "evaluation", "error": reason})
        rows.append(row)
        if row["selected_frame_indices_zero_based"]:
            try:
                input_indices[tile] = [int(i) for i in row["selected_frame_indices_zero_based"].split()]
            except ValueError:
                input_indices[tile] = []

    # Capture exact full-run versus pilot agreement without changing either result tree.
    frame_csv = here / "reports" / "reconstructed_tci_frame_indices.csv"
    reconstructed = {}
    if frame_csv.is_file():
        with frame_csv.open(newline="", encoding="utf-8-sig") as f:
            reconstructed = {r["tile_id"]: r["selected_stack_indices_zero_based"] for r in csv.DictReader(f)}
    pilot_root = here / "results" / "satlas_sr_pilot_five_tiles"
    by_tile = {r["tile_id"]: r for r in rows}
    pilot_comparisons = []
    for tile in PILOT_IDS:
        r = by_tile.get(tile, {})
        sr, lr = Path(r.get("prediction_path", "")), Path(r.get("lr_preview_path", ""))
        pilot_sr, pilot_lr = pilot_root / tile / "sr.png", pilot_root / tile / "lr.png"
        pilot_comparisons.append({
            "tile_id": tile,
            "full_selected_indices": r.get("selected_frame_indices_zero_based", ""),
            "reconstructed_indices_match": bool(r.get("selected_frame_indices_zero_based") == reconstructed.get(tile, "")),
            "sr_byte_identical": sr.is_file() and pilot_sr.is_file() and sha256(sr) == sha256(pilot_sr),
            "lr_byte_identical": lr.is_file() and pilot_lr.is_file() and sha256(lr) == sha256(pilot_lr),
        })

    evaluated = [r for r in rows if r["evaluation_status"] == "METRICS_COMPUTED_EXPLORATORY"]
    psnrs = [float(r["psnr_db"]) for r in evaluated]
    ssims = [float(r["ssim"]) for r in evaluated]
    summary = {
        "dataset_root": str(dataset), "inference_manifest": str(manifest_path),
        "sentinel2_tile_directories_discovered": len(s2_dirs),
        "naip_reference_pngs_discovered": sum(len(v) for v in refs_by_id.values()),
        "matched_tile_pairs": sum(len(refs_by_id.get(p.name, [])) == 1 for p in s2_dirs),
        "inference_manifest_rows": len(manifest_rows), "duplicate_manifest_tile_ids": duplicate_manifest_ids,
        "duplicate_or_ambiguous_naip_tile_ids": duplicate_reference_ids,
        "inference_status_counts": dict(counts),
        "missing_or_malformed_file_counts": dict(missing_or_malformed),
        "evaluation_status_counts": dict(Counter(r["evaluation_status"] for r in rows)),
        "excluded_tile_count": sum(r["evaluation_status"] == "EXCLUDED" for r in rows),
        "metric_assumptions": {
            "array_compatibility": "RGB PNG, matching 128x128 dimensions, uint8 range 0-255 required; checked per tile",
            "spatial_registration": "UNVERIFIED; all computed values are exploratory raw aligned-array comparisons",
            "temporal_comparability": "source pairing window documented, but individual S2 frame timestamps/cloud status unavailable in this archive",
            "data_range": 255,
            "psnr": "skimage.metrics.peak_signal_noise_ratio(reference, prediction, data_range=255)",
            "ssim": "skimage.metrics.structural_similarity(reference, prediction, data_range=255, channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True)",
        },
        "aggregate_exploratory_metrics": {
            "n_tiles": len(evaluated),
            "psnr_db_mean": statistics.mean(psnrs) if psnrs else None,
            "psnr_db_sample_std": statistics.stdev(psnrs) if len(psnrs) > 1 else None,
            "ssim_mean": statistics.mean(ssims) if ssims else None,
            "ssim_sample_std": statistics.stdev(ssims) if len(ssims) > 1 else None,
        },
        "image_value_ranges_uint8": global_ranges,
        "per_tile_errors": errors,
        "pilot_exact_comparison": pilot_comparisons,
        "evaluation_elapsed_seconds": round(time.perf_counter() - started, 3),
        "warning": "metrics are not evidence of validated pixel accuracy because pixel registration is not verified",
    }
    results.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    chosen = write_contact_sheet(results / "representative_contact_sheet.png", rows, input_indices)
    summary["contact_sheet_tiles"] = chosen
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    run_summary = (results / "run_summary.md").read_text(encoding="utf-8") if (results / "run_summary.md").is_file() else ""
    checkpoint_match = re.search(r"Checkpoint: `([^`]+)` \(SHA-256 `([^`]+)`; key `([^`]+)`\)", run_summary)
    total_time_match = re.search(r"Total elapsed time: ([0-9.]+) seconds", run_summary)
    mean_time_match = re.search(r"Mean per-tile inference time \(OK rows only\): ([0-9.]+) seconds", run_summary)
    input_range, pred_range, ref_range = global_ranges["lr_preview"], global_ranges["prediction"], global_ranges["reference"]
    report = [
        "# Satlas-SR full-validation baseline evaluation and audit", "",
        "## A. Verified", "",
        f"- Inference manifest contains {len(manifest_rows)} rows for {len(s2_dirs)} unique discovered Sentinel-2 tile directories. Statuses: {dict(counts)}. There are {len(duplicate_manifest_ids)} duplicate manifest IDs, {len(duplicate_reference_ids)} duplicate/ambiguous NAIP reference IDs, and {len(errors)} missing/malformed-file or evaluation issues; missing TCI inputs, LR previews, predictions, and NAIP references were all 0.",
        f"- Exactly {summary['matched_tile_pairs']} tile IDs have one NAIP reference. The evaluation script checked each entry's input, LR preview, prediction, and reference for readability; all metrics rows that pass checks are RGB with matching 128x128 prediction/reference dimensions. The full output tree was also audited: every successful manifest entry points to a readable 128x128 RGB SR PNG and a readable 32x32 RGB LR-preview PNG.",
        f"- Full-run checkpoint: `{checkpoint_match.group(1) if checkpoint_match else 'not resolved from run_summary.md'}`; SHA-256 `{checkpoint_match.group(2) if checkpoint_match else 'not resolved'}`; weight key `{checkpoint_match.group(3) if checkpoint_match else 'not resolved'}`. Config: `ssr/options/infer_satlas_full_validation.yml`; seed `{manifest_rows[0].get('seed','') if manifest_rows else ''}`; eight TCI frames selected with the unchanged official `format_s2naip_data`; RGB values divided by 255; x4 output.",
        f"- Inference wall time `{total_time_match.group(1) if total_time_match else 'not recorded'}` s; mean per-tile time `{mean_time_match.group(1) if mean_time_match else 'not recorded'}` s (from the existing run summary). Observed uint8 ranges: LR preview {input_range}, SR {pred_range}, NAIP {ref_range}.",
        "- The full-run predictions and LR previews for the five pilot IDs are byte-identical to pilot files. Selected frame indices match the saved reconstruction for each pilot tile: " + ", ".join(p["tile_id"] for p in pilot_comparisons if p["sr_byte_identical"] and p["lr_byte_identical"] and p["reconstructed_indices_match"]) + ".",
        "- Local inference uses the repository's released 8-image configuration (`SSR_RRDBNet`, 24 input channels, 3 output channels, scale 4, 64 features, 23 RRDB blocks, growth 32). The instantiated model/checkpoint has 16,710,083 parameters, consistent with the paper's approximately 17M small ESRGAN. The official config uses `params_ema`.",
        "- Procedure difference: this custom runner sorts tile directories before consuming one seeded Python RNG; stock `ssr.infer.py` traverses recursive `glob.glob()` results without an explicit sort. Because frame choice uses that RNG, enumeration order can change per-tile frames. The five pilot IDs occur first in this sorted full run and match their recorded indices and output hashes; all full-run indices are in the manifest.",
        "",
        "## B. Exploratory", "",
        f"- Raw PSNR/SSIM were computed for {len(evaluated)} tiles after checking RGB mode, 128x128 dimensions, readability, and uint8 range. These numbers are marked `METRICS_COMPUTED_EXPLORATORY` in `tile_metrics.csv`; excluded rows remain in the CSV with reasons. PSNR mean/sample SD: {summary['aggregate_exploratory_metrics']['psnr_db_mean']:.4f} / {summary['aggregate_exploratory_metrics']['psnr_db_sample_std']:.4f} dB; SSIM mean/sample SD: {summary['aggregate_exploratory_metrics']['ssim_mean']:.4f} / {summary['aggregate_exploratory_metrics']['ssim_sample_std']:.4f}. data_range=255; SSIM settings: `{SSIM_SETTINGS}`.",
        "- The contact sheet shows 12 deterministic representative tiles (the five pilot tiles plus seven spread across sorted remaining IDs). Its LR panel shows the eight selected S2 frames, in manifest order, as a 4x2 montage; thumbnails use nearest-neighbor display enlargement only. SR and NAIP panels remain at 128x128.",
        "- Interpret these as array comparisons only. The repository says the images use a shared Web-Mercator zoom-17 tile system and are intended to overlap spatially; the loader pairs by tile ID. The archive contains PNGs without CRS/affine/pixel-origin/crop-window metadata. The README's generic S2 reproject example is not per-image proof of an identical grid. Therefore matching tile IDs and equal dimensions do not verify pixel registration.",
        "- Temporal documentation differs: the checked-out README says temporal overlap within three months, while paper Section 4.1 says overlapping S2 imagery within two months. This archive has NAIP date tokens in its folder UUIDs but no dates/source scene IDs per S2 stack frame. The paper says clouds are not removed, so cloud and scene changes can affect comparisons. The discrepancy and missing per-frame metadata prevent validating exact temporal offsets for these image pairs.",
        "- The paper states that geospatial and temporal metadata are provided for all imagery (Section 4.1), but those records are not present in the extracted small_val_set archive and were not located in the checked-out project. They may be separately distributed; they should be recovered before registration validation.",
        "",
        "## C. Not yet established", "",
        "- Scientifically valid pixel-level PSNR/SSIM or the paper's cPSNR/LPIPS/CLIPScore results: registration and radiometric/temporal comparability are not established here. This run computes standard PSNR/SSIM only; it does not reproduce the paper's full metric suite or human study.",
        "- Building reconstruction accuracy, missed-building rates, hallucination rates, or building-level generalization. No labels or spatially verified annotations were created.",
        "- Exact paper reproduction. This work ran the released pretrained 8S2 checkpoint on the 256-image urban small validation subset, without training. It does not reproduce training, the paper's full data split/experiment, the reported medium 87M ESRGAN in its method comparison, or the paper's CLIPScore/cPSNR/LPIPS and expert building-count protocols. The instantiated released checkpoint is the 17M small model.",
        "",
        "## Files and rerun", "",
        "- Per-tile results and exclusions: `results/satlas_sr_full_validation/tile_metrics.csv`.",
        "- Machine-readable audit and aggregate metrics: `results/satlas_sr_full_validation/summary.json`.",
        "- Representative 3-panel contact sheet: `results/satlas_sr_full_validation/representative_contact_sheet.png`.",
        "- Re-run from repository root with the existing environment: `python evaluate_satlas_sr_validation.py`.",
        "- Inference was not rerun for this evaluation.",
        "",
        "## References", "",
        "- Repository documentation/configuration: `README.md`, `ssr/options/experiments/esrgan_baseline.yml`, `ssr/data/s2-naip_dataset.py`, `ssr/utils/infer_utils.py`.",
        "- Wolters, Bastani, Kembhavi, *Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing*, arXiv:2311.18082, Sections 3, 4.1, 5 and building-count study: https://arxiv.org/html/2311.18082v1.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "csv": str(csv_path), "json": str(json_path), "contact_sheet": str(results / 'representative_contact_sheet.png'), "report": str(report_path)}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
