"""Evaluate five Satlas-SR outputs against their tile-matched NAIP references."""
import argparse
import csv
import statistics
import warnings
from pathlib import Path

from PIL import Image, ImageDraw

TILE_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]
CSV_FIELDS = ["row_type", "tile_id", "status", "psnr_db", "ssim", "n_tiles", "failed_tiles",
              "data_range", "ssim_settings", "alignment_check", "warning", "error"]
SSIM_SETTINGS = "channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True"
ALIGNMENT_NOTE = "same tile ID and equal 128x128 dimensions; PNGs have no georeferencing, pixel registration is assumed, not verified"


def image_rgb(path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"expected PNG, got {im.format}")
        if im.mode != "RGB":
            raise ValueError(f"expected RGB mode, got {im.mode}")
        if im.size != (128, 128):
            raise ValueError(f"expected 128x128, got {im.size}")
        return im.copy()


def make_contact_sheet(path, tiles, panels):
    margin, gap, label_h, image_size, row_gap = 12, 14, 28, 128, 14
    width = margin * 2 + image_size * 2 + gap
    row_height = label_h + image_size + row_gap
    height = margin * 2 + row_height * len(tiles)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, tile in enumerate(tiles):
        y = margin + idx * row_height
        draw.text((margin, y), f"{tile}  |  ESRGAN output                         NAIP reference", fill="black")
        panel_y = y + label_h
        pred, ref = panels.get(tile, (None, None))
        for x, image, label in ((margin, pred, "ESRGAN unavailable"), (margin + image_size + gap, ref, "NAIP unavailable")):
            if image is not None:
                sheet.paste(image, (x, panel_y))
            else:
                draw.rectangle((x, panel_y, x + image_size - 1, panel_y + image_size - 1), outline="red", width=2)
                draw.text((x + 7, panel_y + 56), label, fill="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--dataset", type=Path, default=here.parent / "data" / "small_val_set")
    parser.add_argument("--results", type=Path, default=here / "results" / "satlas_sr_pilot_five_tiles")
    args = parser.parse_args()
    dataset, results = args.dataset.resolve(), args.results.resolve()
    log_path = results / "inference_log.csv"
    inference = {}
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8-sig") as handle:
            inference = {row["tile_id"]: row for row in csv.DictReader(handle)}

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import numpy as np

    rows, panels, good = [], {}, []
    for tile in TILE_IDS:
        row = {key: "" for key in CSV_FIELDS}
        row.update(row_type="TILE", tile_id=tile, status="FAILED", data_range="255",
                   ssim_settings=SSIM_SETTINGS, alignment_check=ALIGNMENT_NOTE)
        pred_img = ref_img = None
        try:
            inference_row = inference.get(tile)
            if not inference_row:
                raise RuntimeError("tile missing from inference_log.csv")
            if inference_row.get("status") != "OK":
                raise RuntimeError(f"inference status {inference_row.get('status')}: {inference_row.get('error', '')}")
            output_path = results / tile / "sr.png"
            matches = [p for p in (dataset / "naip").rglob("rgb.png") if p.parent.name == tile]
            if len(matches) != 1:
                raise RuntimeError(f"expected exactly one NAIP rgb.png for tile ID, found {len(matches)}")
            pred_img, ref_img = image_rgb(output_path), image_rgb(matches[0])
            if pred_img.size != ref_img.size:
                raise ValueError(f"output/reference dimensions differ: {pred_img.size} vs {ref_img.size}")
            pred = np.asarray(pred_img, dtype=np.uint8)
            ref = np.asarray(ref_img, dtype=np.uint8)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                row["psnr_db"] = f"{peak_signal_noise_ratio(ref, pred, data_range=255):.8f}"
                row["ssim"] = f"{structural_similarity(ref, pred, data_range=255, channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True):.8f}"
            row["warning"] = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)
            row["status"] = "OK"
            good.append((tile, float(row["psnr_db"]), float(row["ssim"])))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        panels[tile] = (pred_img, ref_img)
        rows.append(row)
        print(f"{tile}: {row['status']} PSNR={row['psnr_db']} SSIM={row['ssim']} error={row['error']}")

    failed = len(TILE_IDS) - len(good)
    complete_status = "COMPLETE" if failed == 0 else "INCOMPLETE"
    for label, fn in (("MEAN", statistics.mean), ("SAMPLE_STD", statistics.stdev)):
        summary = {key: "" for key in CSV_FIELDS}
        summary.update(row_type=label, tile_id="ALL_FIVE", status=complete_status,
                       n_tiles=len(good), failed_tiles=failed, data_range="255",
                       ssim_settings=SSIM_SETTINGS, alignment_check=ALIGNMENT_NOTE)
        if good and (label == "MEAN" or len(good) > 1):
            summary["psnr_db"] = f"{fn([item[1] for item in good]):.8f}"
            summary["ssim"] = f"{fn([item[2] for item in good]):.8f}"
        if failed:
            summary["error"] = f"Aggregate uses {len(good)} successful tile(s); {failed} tile(s) failed and are retained as rows above."
        rows.append(summary)

    csv_path = results / "five_tile_metrics.csv"
    results.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sheet_path = results / "five_tile_contact_sheet.png"
    make_contact_sheet(sheet_path, TILE_IDS, panels)
    print(f"CSV: {csv_path}")
    print(f"Contact sheet: {sheet_path}")
    print(f"Successful tiles: {len(good)}/5; failed tiles: {failed}")
    print("Metric interpretation requires spatial alignment and comparable acquisition conditions; PNG metadata does not verify registration.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())