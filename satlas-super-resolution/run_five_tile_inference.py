"""Run the five-tile Satlas-SR pilot with the repository's official preprocessing helper."""
import argparse
import csv
import random
import time
import warnings
from pathlib import Path

TILE_IDS = ["20771_46525", "20890_46801", "20926_46045", "21068_45692", "21856_49804"]
FIELDS = ["tile_id", "status", "runtime_seconds", "gpu_peak_allocated_mib", "input_path", "output_path", "error"]


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", required=True, help="Five-tile inference YAML configuration")
    args = parser.parse_args()
    rows = [{"tile_id": tile, "status": "NOT_RUN", "runtime_seconds": "", "gpu_peak_allocated_mib": "",
             "input_path": "", "output_path": "", "error": ""} for tile in TILE_IDS]
    cfg_path = Path(args.opt).resolve()
    log_path = None
    try:
        with warnings.catch_warnings(record=True) as import_caught:
            warnings.simplefilter('always')
            from ssr.utils.options import yaml_load
            opt = yaml_load(str(cfg_path))
        data_root = Path(opt["data_dir"])
        save_root = Path(opt["save_path"])
        log_path = save_root / "inference_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        random.seed(int(opt.get("manual_seed", 0)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import numpy as np
            import skimage.io
            import torch
            from ssr.utils.infer_utils import format_s2naip_data
            from ssr.utils.model_utils import build_network

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; GPU inference was required")
            device = torch.device("cuda")
            model = build_network(opt)
            checkpoint = torch.load(opt["path"]["pretrain_network_g"], map_location="cpu")
            key = opt["path"]["param_key_g"]
            model.load_state_dict(checkpoint[key], strict=bool(opt["path"]["strict_load_g"]))
            model = model.to(device).eval()

            for row in rows:
                tile = row["tile_id"]
                input_path = data_root / tile / "tci.png"
                tile_out = save_root / tile
                sr_path = tile_out / "sr.png"
                row["input_path"] = str(input_path)
                row["output_path"] = str(sr_path)
                started = time.perf_counter()
                try:
                    if not input_path.is_file():
                        raise FileNotFoundError(f"Missing TCI input: {input_path}")
                    image = skimage.io.imread(str(input_path))
                    input_tensor, low_res_image = format_s2naip_data(image, int(opt["n_lr_images"]), device)
                    torch.cuda.reset_peak_memory_stats(device)
                    with torch.inference_mode():
                        prediction = model(input_tensor)
                    torch.cuda.synchronize(device)
                    prediction = torch.clamp(prediction, 0, 1).squeeze().cpu().numpy()
                    prediction = np.transpose(prediction, (1, 2, 0))
                    prediction = (prediction * 255).astype(np.uint8)
                    tile_out.mkdir(parents=True, exist_ok=True)
                    skimage.io.imsave(str(tile_out / "lr.png"), low_res_image)
                    skimage.io.imsave(str(sr_path), prediction, check_contrast=False)
                    row["gpu_peak_allocated_mib"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f}"
                    row["status"] = "OK"
                except Exception as exc:
                    row["status"] = "FAILED"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    row["runtime_seconds"] = f"{time.perf_counter() - started:.3f}"
                    write_log(log_path, rows)

        warning_lines = [f"{item.category.__name__}: {item.message}" for item in import_caught] + [f"{item.category.__name__}: {item.message}" for item in caught]
        warning_path = save_root / "inference_warnings.txt"
        warning_path.write_text("\n".join(warning_lines) + ("\n" if warning_lines else "No warnings.\n"), encoding="utf-8")
        for line in warning_lines:
            print("WARNING:", line)
    except Exception as exc:
        message = f"Setup failure: {type(exc).__name__}: {exc}"
        for row in rows:
            if row["status"] == "NOT_RUN":
                row["status"] = "FAILED"
                row["error"] = message
        if log_path is None:
            log_path = cfg_path.parent.parent.parent / "results" / "satlas_sr_pilot_five_tiles" / "inference_log.csv"
        write_log(log_path, rows)
        print(message)

    for row in rows:
        print(f"{row['tile_id']}: {row['status']} runtime={row['runtime_seconds']}s output={row['output_path']} error={row['error']}")
    return 1 if any(row["status"] != "OK" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())