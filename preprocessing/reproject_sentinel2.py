#!/usr/bin/env python3
"""
Sentinel-2 raw image reprojection and tiling pipeline script.
Implements bilinear reprojection to EPSG:3857 (Web-Mercator) at 9.555m resolution
matching the official S2-NAIP dataset format described in Section 4.1 of the Satlas-SR paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def reproject_and_tile_sentinel2(
    input_jp2_path: Path,
    output_dir: Path,
    target_epsg: int = 3857,
    resolution_m: float = 9.555,
) -> None:
    """
    Reprojects an ESA Sentinel-2 L1C product (TCI or raw band) into Web-Mercator
    and tiles into the 32x32 pixel chip representation.
    Requires rasterio / gdal.
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError:
        raise ImportError(
            "rasterio is required for raw Sentinel-2 reprojection. "
            "Install with: conda install -c conda-forge rasterio gdal"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dst_crs = f"EPSG:{target_epsg}"

    with rasterio.open(input_jp2_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=(resolution_m, resolution_m)
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        reprojected_path = output_dir / f"{input_jp2_path.stem}_reprojected.tif"
        with rasterio.open(reprojected_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

        print(f"Reprojected raw raster saved to: {reprojected_path}")
        print(f"Output shape: {width} x {height} at {resolution_m}m GSD")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jp2", type=Path, required=True, help="Path to Sentinel-2 TCI.jp2 or band file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save reprojected raster")
    parser.add_argument("--resolution", type=float, default=9.555, help="Target ground resolution in meters (default: 9.555)")
    args = parser.parse_args()

    reproject_and_tile_sentinel2(args.input_jp2, args.output_dir, resolution_m=args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

