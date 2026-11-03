#!/usr/bin/env python3
"""
Data inspection tool for Satlas-SR datasets.
Verifies dimensions, channels, bit-depth, temporal stack heights, and readable integrity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


def inspect_dataset(dataset_root: Path, sample_limit: int = 2) -> dict:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    s2_root = root / "sentinel2"
    naip_root = root / "naip"
    if not s2_root.is_dir() or not naip_root.is_dir():
        raise FileNotFoundError(f"Expected sentinel2/ and naip/ under {root}")

    s2_pngs = sorted(s2_root.rglob("*.png"))
    naip_pngs = sorted(naip_root.rglob("*.png"))
    s2_folders = sorted(p for p in s2_root.iterdir() if p.is_dir())
    naip_refs = sorted(naip_root.rglob("rgb.png"))
    naip_sample_ids = {p.parent.name for p in naip_refs}
    s2_ids = {p.name for p in s2_folders}

    print(f"Dataset root: {root}")
    print(f"Total PNG files: {len(s2_pngs) + len(naip_pngs)}")
    print(f"Sentinel-2 PNG stacks: {len(s2_pngs)}")
    print(f"Sentinel-2 tile folders: {len(s2_folders)}")
    print(f"NAIP PNG files: {len(naip_pngs)}")
    print(f"NAIP reference chips (rgb.png): {len(naip_refs)}")
    print(f"Matching S2 & NAIP tile IDs: {len(s2_ids & naip_sample_ids)}")
    print(f"S2-only / NAIP-only tile IDs: {len(s2_ids - naip_sample_ids)} / {len(naip_sample_ids - s2_ids)}")

    errors = []
    modality_stats = {}
    for label, paths in (("sentinel2", s2_pngs), ("naip", naip_pngs)):
        stats = Counter()
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    stats[(image.size, image.mode)] += 1
            except Exception as exc:
                errors.append((path, exc))
        modality_stats[label] = stats
        print(f"\n{label} dimensions/modes (width, height, mode: count):")
        for (size, mode), count in sorted(stats.items(), key=lambda item: (item[0][0], item[0][1])):
            print(f"  {size} {mode}: {count}")

    frame_layouts = Counter()
    for path in s2_pngs:
        try:
            with Image.open(path) as image:
                width, height = image.size
            if width == 32 and height % 32 == 0:
                frame_layouts[height // 32] += 1
        except Exception:
            pass
    print(f"\nSentinel-2 frame stack distribution (revisit frames: stack count):")
    for frames, count in sorted(frame_layouts.items()):
        print(f"  {frames} frames ({frames * 32}px height): {count} stacks")

    if sample_limit > 0:
        print(f"\nSample inspection (first {sample_limit} tiles):")
        for folder in s2_folders[:sample_limit]:
            files = sorted(folder.glob("*.png"))
            print(f"  Tile {folder.name}: {len(files)} band stacks")
            for path in files:
                try:
                    with Image.open(path) as image:
                        print(f"    - {path.name}: {image.size} {image.mode}")
                except Exception as exc:
                    print(f"    - ERROR {path.name}: {exc}")
            refs = [p for p in naip_refs if p.parent.name == folder.name]
            for path in refs:
                try:
                    with Image.open(path) as image:
                        print(f"    - NAIP {path.name}: {image.size} {image.mode}")
                except Exception as exc:
                    print(f"    - ERROR {path}: {exc}")

    if errors:
        print(f"\nUnreadable / Corrupt PNG files: {len(errors)}")
        for path, exc in errors:
            print(f"  {path}: {exc}")

    return {
        "total_s2_tiles": len(s2_folders),
        "total_naip_refs": len(naip_refs),
        "matching_tiles": len(s2_ids & naip_sample_ids),
        "errors": len(errors),
    }


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent / "data" / "small_val_set"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=default_root,
                        help=f"Dataset root path (default: {default_root})")
    parser.add_argument("--samples", type=int, default=2,
                        help="Number of sample tiles to display (default: 2)")
    args = parser.parse_args()

    results = inspect_dataset(args.dataset, sample_limit=args.samples)
    return 1 if results["errors"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

