#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按帧顺序读取 image + mask，并把每个 mask 单独抠图保存为一张 PNG。

关键逻辑：
- image 不是按名字匹配
- 而是按排序后的顺序匹配
- 如果 mask 在 masks_root/3/ 下，就取 image_root 里排序后的第 4 张图片

目录示例：

image_root/
  xxx_a.jpg
  foo.png
  bar.jpg
  ...

masks_root/
  0/
    instance_0001.png
    instance_0002.png
  1/
    instance_0003.png
  3/
    instance_0004.png

输出：
output_dir/
  0/
    instance_0001.png
    instance_0002.png
  1/
    instance_0003.png
  3/
    instance_0004.png
"""

import re
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def cut_single_instance(image: np.ndarray, mask: np.ndarray, bbox_only: bool) -> np.ndarray | None:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image shape {image.shape[:2]} != mask shape {mask.shape[:2]}")

    if not np.any(mask):
        return None

    # ✅ 黑背景版本
    rgb = np.zeros_like(image)
    rgb[mask] = image[mask]

    if bbox_only:
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        rgb = rgb[y0:y1 + 1, x0:x1 + 1]

    return rgb


def collect_sorted_images(image_root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        [p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: natural_key(p.name)
    )
    return image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="原图目录")
    parser.add_argument("--masks_root", type=str, required=True, help="mask根目录，每帧一个子目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--bbox_only", action="store_true", help="是否只保存紧致bbox区域")
    parser.add_argument("--skip_empty_mask", action="store_true", help="空mask时跳过；否则报错")
    args = parser.parse_args()

    image_root = Path(args.image_dir)
    masks_root = Path(args.masks_root)
    output_dir = Path(args.output_dir)

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_sorted_images(image_root)
    if not image_files:
        raise FileNotFoundError(f"No images found under: {image_root}")

    frame_dirs = sorted(
        [p for p in masks_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)],
        key=lambda p: natural_key(p.name)
    )

    total_masks = 0
    saved_masks = 0

    print(f"[INFO] total images: {len(image_files)}")
    print(f"[INFO] total frame dirs: {len(frame_dirs)}")

    for frame_dir in tqdm(frame_dirs, desc="Processing frames"):
        frame_id = int(frame_dir.name)

        if frame_id < 0 or frame_id >= len(image_files):
            print(f"[WARN] frame_id {frame_id} out of range, image count = {len(image_files)}")
            continue

        img_path = image_files[frame_id]
        image = load_rgb_image(img_path)

        mask_files = sorted(
            [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]],
            key=lambda p: natural_key(p.name)
        )

        frame_out_dir = output_dir / frame_dir.name
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        for mask_path in mask_files:
            total_masks += 1
            mask = load_binary_mask(mask_path)

            if not np.any(mask):
                msg = f"[WARN] empty mask: {mask_path}"
                if args.skip_empty_mask:
                    print(msg)
                    continue
                raise ValueError(msg)

            rgba = cut_single_instance(image, mask, bbox_only=args.bbox_only)
            if rgba is None:
                if args.skip_empty_mask:
                    continue
                raise RuntimeError(f"Failed to cut instance for mask: {mask_path}")

            out_path = frame_out_dir / f"{mask_path.stem}.png"
            Image.fromarray(rgba).save(out_path)
            saved_masks += 1

    print(f"[INFO] total mask files: {total_masks}")
    print(f"[INFO] saved png files: {saved_masks}")

    if saved_masks != total_masks:
        print("[WARN] saved count != mask count. Usually caused by empty mask or frame index out of range.")
    else:
        print("[INFO] saved count == mask count")


if __name__ == "__main__":
    main()