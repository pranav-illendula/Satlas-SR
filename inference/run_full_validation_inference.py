"""Reproducible, resumable inference over the Satlas-SR small validation set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from PIL import Image

FIELDS = [
    "tile_id", "status", "selected_frame_indices_zero_based", "seed", "frame_policy",
    "input_path", "naip_reference_path", "sr_output_path", "lr_output_path",
    "input_sha256", "checkpoint_sha256", "config_sha256", "runtime_seconds",
    "gpu_peak_allocated_mib", "error",
]
PILOT_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]
FRAME_POLICY = "sorted tile IDs; one global Python random stream seeded once; official format_s2naip_data random.sample selection"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def image_is_valid(path: Path, expected_size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "PNG" and image.mode == "RGB" and image.size == expected_size
    except Exception:
        return False


def discover(s2_root: Path, naip_root: Path):
    s2_dirs = sorted((p for p in s2_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    ref_map: dict[str, list[Path]] = {}
    for ref in naip_root.rglob("rgb.png"):
        ref_map.setdefault(ref.parent.name, []).append(ref)
    return s2_dirs, ref_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("-opt", required=True, help="YAML inference config")
    parser.add_argument("--tiles", nargs="+", help="optional tile IDs for a subset/smoke run")
    parser.add_argument("--output-dir", type=Path, help="override configured save_path")
    parser.add_argument("--force-new-run", action="store_true",
                        help="use a fresh timestamped output directory, preserving previous results")
    args = parser.parse_args()

    from ssr.utils.options import yaml_load
    opt_path = Path(args.opt).resolve()
    opt = yaml_load(str(opt_path))
    s2_root = Path(opt["data_dir"]).resolve()
    naip_root = Path(opt["naip_dir"]).resolve()
    checkpoint_path = Path(opt["path"]["pretrain_network_g"]).resolve()
    output_root = (args.output_dir or Path(opt["save_path"])).resolve()
    seed = int(opt.get("manual_seed", 0))

    if args.force_new_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = output_root.with_name(output_root.name + "_rerun_" + stamp)
    if not s2_root.is_dir() or not naip_root.is_dir():
        raise FileNotFoundError(f"Expected Sentinel-2 and NAIP directories: {s2_root}; {naip_root}")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {checkpoint_path}")

    s2_dirs, ref_map = discover(s2_root, naip_root)
    selected_ids = set(args.tiles) if args.tiles else None
    if selected_ids:
        unknown = selected_ids - {p.name for p in s2_dirs}
        if unknown:
            raise ValueError("Requested tile IDs not found under Sentinel-2 root: " + ", ".join(sorted(unknown)))
        s2_dirs = [p for p in s2_dirs if p.name in selected_ids]
    chosen_ids = [p.name for p in s2_dirs]
    config_record = {
        "config": str(opt_path), "seed": seed, "tile_ids_in_order": chosen_ids,
        "n_lr_images": int(opt["n_lr_images"]), "scale": int(opt["scale"]),
        "network_g": opt["network_g"], "checkpoint": str(checkpoint_path),
        "param_key_g": opt["path"]["param_key_g"], "strict_load_g": bool(opt["path"]["strict_load_g"]),
        "frame_policy": FRAME_POLICY,
    }
    config_sha = hashlib.sha256(json.dumps(config_record, sort_keys=True).encode("utf-8")).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "inference_manifest.csv"
    old_rows = {}
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            old_rows = {row.get("tile_id", ""): row for row in csv.DictReader(handle)}

    rows = []
    work = []
    for tile_dir in s2_dirs:
        tile = tile_dir.name
        refs = ref_map.get(tile, [])
        row = {field: "" for field in FIELDS}
        row.update(tile_id=tile, status="PENDING", seed=seed, frame_policy=FRAME_POLICY,
                   input_path=str(tile_dir / "tci.png"),
                   naip_reference_path=str(refs[0]) if len(refs) == 1 else "",
                   sr_output_path=str(output_root / tile / "sr.png"),
                   lr_output_path=str(output_root / tile / "lr.png"), config_sha256=config_sha)
        if len(refs) != 1:
            row.update(status="FAILED", error=f"Expected exactly one NAIP rgb.png for tile ID, found {len(refs)}")
        elif not (tile_dir / "tci.png").is_file():
            row.update(status="FAILED", error=f"Missing TCI input: {tile_dir / 'tci.png'}")
        else:
            work.append(row)
        rows.append(row)
    atomic_csv(manifest_path, rows)

    # Fail clearly for missing runtime/model setup; never substitute another device or model.
    import numpy as np
    import skimage.io
    import torch
    import ssr.utils.infer_utils as infer_utils
    from ssr.utils.model_utils import build_network

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this run requires the configured GPU")
    device = torch.device("cuda")
    model = build_network(opt)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    parameter_key = opt["path"]["param_key_g"]
    if parameter_key not in checkpoint:
        raise KeyError(f"Checkpoint lacks configured parameter key {parameter_key!r}; keys={list(checkpoint.keys())}")
    model.load_state_dict(checkpoint[parameter_key], strict=bool(opt["path"]["strict_load_g"]))
    del checkpoint
    model = model.to(device).eval()
    checkpoint_hash = sha256(checkpoint_path)
    for row in rows:
        row["checkpoint_sha256"] = checkpoint_hash
    atomic_csv(manifest_path, rows)

    # Preserve the official formatter. This wrapper only observes the exact sample call,
    # including when replaying a valid completed tile to maintain RNG state on resume.
    original_sample = infer_utils.random.sample
    current_selection = []
    def recording_sample(population, k):
        chosen = original_sample(population, k)
        current_selection.extend(chosen)
        return chosen
    infer_utils.random.sample = recording_sample

    random.seed(seed)
    start_all = time.perf_counter()
    try:
        for row in work:
            started = time.perf_counter()
            input_path = Path(row["input_path"])
            tile_output = Path(row["sr_output_path"]).parent
            try:
                input_hash = sha256(input_path)
                previous = old_rows.get(row["tile_id"], {})
                image = skimage.io.imread(str(input_path))
                if image.ndim != 3 or image.shape[1] != 32 or image.shape[2] != 3 or image.shape[0] % 32:
                    raise ValueError(f"Expected stacked RGB TCI shape (32*n,32,3), got {image.shape}")
                current_selection.clear()
                input_tensor, low_res_image = infer_utils.format_s2naip_data(image, int(opt["n_lr_images"]), device)
                frame_indices = " ".join(str(int(index)) for index in current_selection)
                row["selected_frame_indices_zero_based"] = frame_indices
                row["input_sha256"] = input_hash

                resumable = (
                    previous.get("status") == "OK"
                    and previous.get("input_sha256") == input_hash
                    and previous.get("checkpoint_sha256") == checkpoint_hash
                    and previous.get("config_sha256") == config_sha
                    and previous.get("selected_frame_indices_zero_based") == frame_indices
                    and image_is_valid(Path(row["sr_output_path"]), (128, 128))
                    and image_is_valid(Path(row["lr_output_path"]), (32, 32))
                )
                if resumable:
                    row["status"] = "SKIPPED_VALID"
                    row["runtime_seconds"] = "0.000"
                    row["gpu_peak_allocated_mib"] = previous.get("gpu_peak_allocated_mib", "")
                    row["error"] = ""
                else:
                    torch.cuda.reset_peak_memory_stats(device)
                    with torch.inference_mode():
                        prediction = model(input_tensor)
                    torch.cuda.synchronize(device)
                    prediction = torch.clamp(prediction, 0, 1).squeeze().cpu().numpy()
                    if prediction.shape != (3, 128, 128):
                        raise ValueError(f"Expected model output (3,128,128), got {prediction.shape}")
                    prediction = np.transpose(prediction, (1, 2, 0))
                    prediction = (prediction * 255).astype(np.uint8)
                    tile_output.mkdir(parents=True, exist_ok=True)
                    skimage.io.imsave(str(Path(row["lr_output_path"])), low_res_image)
                    skimage.io.imsave(str(Path(row["sr_output_path"])), prediction, check_contrast=False)
                    if not image_is_valid(Path(row["sr_output_path"]), (128, 128)) or not image_is_valid(Path(row["lr_output_path"]), (32, 32)):
                        raise ValueError("Written output failed PNG/RGB/128x128 validation")
                    row["status"] = "OK"
                    row["runtime_seconds"] = f"{time.perf_counter() - started:.3f}"
                    row["gpu_peak_allocated_mib"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f}"
                    row["error"] = ""
            except Exception as exc:
                row["status"] = "FAILED"
                row["runtime_seconds"] = f"{time.perf_counter() - started:.3f}"
                row["error"] = f"{type(exc).__name__}: {exc}"
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            atomic_csv(manifest_path, rows)
            print(f"{row['tile_id']}: {row['status']} frames={row['selected_frame_indices_zero_based']} time={row['runtime_seconds']}s error={row['error']}", flush=True)
    finally:
        infer_utils.random.sample = original_sample

    elapsed = time.perf_counter() - start_all
    counts = {status: sum(row["status"] == status for row in rows) for status in ("OK", "SKIPPED_VALID", "FAILED", "PENDING")}
    matched_count = sum(len(ref_map.get(p.name, [])) == 1 for p in s2_dirs)
    gpu_name = torch.cuda.get_device_name(device)
    timings = [float(row["runtime_seconds"]) for row in rows if row["runtime_seconds"] and row["status"] in ("OK", "SKIPPED_VALID")]
    summary = [
        "# Satlas-SR full-validation inference run",
        "",
        f"- Run completed (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Config: `{opt_path}`",
        f"- Dataset: `{s2_root}`; NAIP refs: `{naip_root}`",
        f"- Checkpoint: `{checkpoint_path}` (SHA-256 `{checkpoint_hash}`; key `{parameter_key}`)",
        f"- Device: {gpu_name}; PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}",
        f"- Discovered in this run: {len(s2_dirs)}; matched to exactly one NAIP reference: {matched_count}",
        f"- Status counts: OK {counts['OK']}, skipped valid {counts['SKIPPED_VALID']}, failed {counts['FAILED']}, pending {counts['PENDING']}",
        f"- Total elapsed time: {elapsed:.3f} seconds",
        f"- Mean per-tile inference time (OK rows only): {statistics.mean([float(r['runtime_seconds']) for r in rows if r['status'] == 'OK' and r['runtime_seconds']]) if any(r['status'] == 'OK' and r['runtime_seconds'] for r in rows) else 0:.3f} seconds",
        f"- Output directory: `{output_root}`",
        f"- Incremental manifest: `{manifest_path}`",
        "",
        "## Reproducibility",
        "",
        f"The official `format_s2naip_data` helper is used unchanged. Python random seed is set once to `{seed}`; tile directories are traversed in sorted ID order, and frame indices are captured by observing the helper's existing `random.sample` calls. Inputs are RGB TCI stacks divided by 255 by the official helper; the configured generator uses 8 S2 frames, 24 input channels, scale 4, and checkpoint key `" + str(parameter_key) + "`. Outputs are clamped to [0,1], converted to uint8 RGB PNGs, and expected to be 128x128. This reproduces the released inference pipeline; it does not verify geospatial registration or temporal alignment and makes no accuracy claim.",
        "",
        "Failures remain in `inference_manifest.csv`; no NAIP-based PSNR/SSIM is calculated by this run.",
    ]
    (output_root / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    return 1 if counts["FAILED"] or counts["PENDING"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
