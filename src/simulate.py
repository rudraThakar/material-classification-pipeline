import argparse
import csv
import json
import os
import time
from glob import glob
from typing import Dict

from infer import infer_torchscript, infer_onnx, load_class_index


def list_images(frames_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for e in exts:
        files.extend(glob(os.path.join(frames_dir, "**", e), recursive=True))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--class_index", type=str, required=True)
    parser.add_argument("--results_csv", type=str, default="results/simulation_log.csv")
    parser.add_argument("--interval_ms", type=int, default=250)
    parser.add_argument("--low_conf_threshold", type=float, default=0.55)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)

    class_index = load_class_index(args.class_index)
    is_onnx = args.model.lower().endswith(".onnx")

    images = list_images(args.frames_dir)
    if not images:
        raise RuntimeError(f"No images found under {args.frames_dir}")

    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_path", "pred_class", "confidence", "low_confidence"])
        for img_path in images:
            t0 = time.time()
            if is_onnx:
                pred_idx, conf = infer_onnx(args.model, img_path, args.img_size)
            else:
                pred_idx, conf = infer_torchscript(args.model, img_path, args.img_size)
            label = class_index.get(pred_idx, str(pred_idx))
            low_flag = conf < args.low_conf_threshold
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.writerow([now, img_path, label, round(conf, 4), low_flag])
            print(f"{now} | {os.path.basename(img_path)} -> {label} ({conf:.2f}){' [LOW]' if low_flag else ''}")
            elapsed = time.time() - t0
            sleep_s = max(0.0, args.interval_ms / 1000.0 - elapsed)
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()


