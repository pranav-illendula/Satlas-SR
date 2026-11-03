#!/usr/bin/env python3
"""
Dataset inventory and audit tool for Satlas-SR.
Verifies paired integrity of Sentinel-2 and NAIP satellite imagery,
checks for missing bands, calculates frame counts, and detects corrupted tiles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_s2_tile(tile_dir: Path) -> dict:
    tci_path = tile_dir / "tci.png"
    record = {
        "tile_id": tile_dir.name,
        "s2_dir": str(tile_dir),
        "tci_exists": tci_path.is_file(),
        "tci_readable": False,
        "tci_format": "",
        "tci_mode": "",
        "tci_dimensions_wh": "",
        "tci_total_frames": 0,
        "tci_valid_frames": 0,
        "tci_partially_black_frames": 0,
        "tci_completely_black_frames": 0,
        "tci_min": None,
        "tci_max": None,
        "other_band_files": [],
        "s2_status": "PENDING",
        "s2_error": "",
    }
    if not record["tci_exists"]:
        record["s2_status"] = "MISSING_TCI"
        record["s2_error"] = "tci.png does not exist"
        return record

    record["other_band_files"] = sorted([p.name for p in tile_dir.glob("*.png") if p.name != "tci.png"])

    try:
        with Image.open(tci_path) as im:
            im.verify()
        with Image.open(tci_path) as im:
            record["tci_readable"] = True
            record["tci_format"] = im.format
            record["tci_mode"] = im.mode
            w, h = im.size
            record["tci_dimensions_wh"] = f"{w}x{h}"
            if w != 32 or h % 32 != 0:
                record["s2_status"] = "INVALID_DIMENSIONS"
                record["s2_error"] = f"Unexpected shape {w}x{h}"
                return record

            total_frames = h // 32
            record["tci_total_frames"] = total_frames
            arr = np.array(im.convert("RGB"))
            record["tci_min"] = int(np.min(arr))
            record["tci_max"] = int(np.max(arr))

            valid_frames = 0
            partial_black = 0
            complete_black = 0
            for idx in range(total_frames):
                frame = arr[idx * 32 : (idx + 1) * 32, :, :]
                black_mask = (frame[:, :, 0] == 0) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 0)
                black_count = int(np.sum(black_mask))
                if black_count == 0:
                    valid_frames += 1
                elif black_count == 32 * 32:
                    complete_black += 1
                else:
                    partial_black += 1

            record["tci_valid_frames"] = valid_frames
            record["tci_partially_black_frames"] = partial_black
            record["tci_completely_black_frames"] = complete_black
            record["s2_status"] = "OK"
    except Exception as exc:
        record["s2_status"] = "CORRUPT_PNG"
        record["s2_error"] = str(exc)

    return record


def audit_dataset_directory(dataset_root: Path) -> dict:
    root = dataset_root.expanduser().resolve()
    s2_root = root / "sentinel2"
    naip_root = root / "naip"

    if not s2_root.is_dir():
        raise FileNotFoundError(f"Missing sentinel2 directory under {root}")

    s2_folders = sorted([p for p in s2_root.iterdir() if p.is_dir()])
    naip_chips = sorted(list(naip_root.rglob("rgb.png"))) if naip_root.is_dir() else []
    naip_by_tile = {}
    for chip in naip_chips:
        naip_by_tile[chip.parent.name] = chip

    print(f"Auditing dataset at: {root}")
    print(f"Sentinel-2 tiles: {len(s2_folders)}")
    print(f"NAIP reference chips: {len(naip_chips)}")

    s2_audits = []
    for folder in s2_folders:
        audit = audit_s2_tile(folder)
        tile_id = folder.name
        audit["has_naip_pair"] = tile_id in naip_by_tile
        audit["naip_path"] = str(naip_by_tile[tile_id]) if tile_id in naip_by_tile else ""
        s2_audits.append(audit)

    ok_count = sum(1 for a in s2_audits if a["s2_status"] == "OK")
    paired_count = sum(1 for a in s2_audits if a.get("has_naip_pair"))

    print(f"Audit completed: {ok_count}/{len(s2_folders)} S2 tiles healthy, {paired_count} paired with NAIP.")
    return {
        "total_s2_tiles": len(s2_folders),
        "healthy_s2_tiles": ok_count,
        "paired_tiles": paired_count,
        "records": s2_audits,
    }


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent / "data" / "small_val_set"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=default_root,
                        help=f"Dataset path to audit (default: {default_root})")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional JSON output path for audit records")
    args = parser.parse_args()

    results = audit_dataset_directory(args.dataset)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Audit report saved to: {args.output_json}")

    return 0 if results["healthy_s2_tiles"] == results["total_s2_tiles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

