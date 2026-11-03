#!/usr/bin/env python3
"""Inspect the extracted Satlas-SR small validation dataset without modifying it."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent / "data" / "small_val_set"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=default_root,
                        help=f"dataset root (default: {default_root})")
    parser.add_argument("--samples", type=int, default=2,
                        help="number of sample folders to show (default: 2)")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    if not root.is_dir():
        parser.error(f"dataset directory does not exist: {root}")

    s2_root, naip_root = root / "sentinel2", root / "naip"
    if not s2_root.is_dir() or not naip_root.is_dir():
        parser.error(f"expected sentinel2/ and naip/ under {root}")

    s2_pngs = sorted(s2_root.rglob("*.png"))
    naip_pngs = sorted(naip_root.rglob("*.png"))
    s2_folders = sorted(p for p in s2_root.iterdir() if p.is_dir())
    naip_refs = sorted(naip_root.rglob("rgb.png"))
    naip_sample_ids = {p.parent.name for p in naip_refs}
    s2_ids = {p.name for p in s2_folders}

    print(f"Dataset root: {root}")
    print(f"PNG files total: {len(s2_pngs) + len(naip_pngs)}")
    print(f"Sentinel-2 PNGs: {len(s2_pngs)}")
    print(f"Sentinel-2 sample folders: {len(s2_folders)}")
    print(f"NAIP PNGs: {len(naip_pngs)}")
    print(f"NAIP reference images (rgb.png): {len(naip_refs)}")
    print(f"NAIP sample folders (unique tile IDs): {len(naip_sample_ids)}")
    print(f"Matching S2/NAIP tile IDs: {len(s2_ids & naip_sample_ids)}")
    print(f"S2-only / NAIP-only tile IDs: {len(s2_ids - naip_sample_ids)} / {len(naip_sample_ids - s2_ids)}")

    per_folder_counts = Counter(len(list(folder.glob("*.png"))) for folder in s2_folders)
    print(f"PNG files per Sentinel-2 folder: {dict(sorted(per_folder_counts.items()))}")
    modality_stats = {}
    errors = []
    for label, paths in (("sentinel2", s2_pngs), ("naip", naip_pngs)):
        stats = Counter()
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    stats[(image.size, image.mode)] += 1
            except Exception as exc:  # report every unreadable image
                errors.append((path, exc))
        modality_stats[label] = stats
        print(f"{label} dimensions/modes (width,height; mode: count):")
        for (size, mode), count in sorted(stats.items(), key=lambda item: (item[0][0], item[0][1])):
            print(f"  {size} {mode}: {count}")

    # Images are stored as one PNG per band. A 32x1024 PNG has 32 vertical
    # 32x32 frames (Pillow reports width before height).
    frame_layouts = Counter()
    for path in s2_pngs:
        try:
            with Image.open(path) as image:
                width, height = image.size
            if width == 32 and height % 32 == 0:
                frame_layouts[height // 32] += 1
        except Exception:
            pass
    print(f"Sentinel-2 vertical 32x32 frame stacks (frames: PNG count): {dict(sorted(frame_layouts.items()))}")

    print("Sample structure:")
    for folder in s2_folders[:args.samples]:
        files = sorted(folder.glob("*.png"))
        print(f"  Sentinel-2 tile {folder.name}: {len(files)} PNGs")
        for path in files:
            try:
                with Image.open(path) as image:
                    print(f"    {path.name}: {image.size} {image.mode}")
            except Exception as exc:
                print(f"    ERROR {path.name}: {exc}")
        refs = [p for p in naip_refs if p.parent.name == folder.name]
        for path in refs:
            try:
                with Image.open(path) as image:
                    print(f"  NAIP reference {path.relative_to(root)}: {image.size} {image.mode}")
            except Exception as exc:
                print(f"  ERROR {path}: {exc}")

    print(f"Unreadable PNGs: {len(errors)}")
    for path, exc in errors:
        print(f"  {path}: {exc}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
