import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def generate_image(size: Tuple[int, int], color: Tuple[int, int, int], noise_level: int = 25) -> Image.Image:
    h, w = size
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :] = np.array(color, dtype=np.uint8)
    noise = np.random.randint(-noise_level, noise_level + 1, size=(h, w, 3))
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--classes", nargs="*", default=["metal", "plastic", "glass", "paper", "fabric"])
    parser.add_argument("--train_per_class", type=int, default=40)
    parser.add_argument("--val_per_class", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"), exist_ok=True)

    rng = np.random.default_rng(42)
    base_colors = rng.integers(low=30, high=225, size=(len(args.classes), 3))

    for idx, cls in enumerate(args.classes):
        for split, n in [("train", args.train_per_class), ("val", args.val_per_class)]:
            cls_dir = os.path.join(args.out_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n):
                # Slightly vary the color per image
                color_jitter = rng.integers(low=-20, high=21, size=(3,))
                color = np.clip(base_colors[idx] + color_jitter, 0, 255).tolist()
                img = generate_image((args.img_size, args.img_size), tuple(int(c) for c in color))
                img.save(os.path.join(cls_dir, f"{cls}_{i:04d}.png"))

    print(f"Dummy dataset created under {args.out_dir} with classes: {', '.join(args.classes)}")


if __name__ == "__main__":
    main()


