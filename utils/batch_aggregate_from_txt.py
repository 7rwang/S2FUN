#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch aggregate masks for specific scene_ids listed in a txt file.

Assumed structure (strict):
  ROOT/<scene_id>/masks/<frame_dir>/*.png

Output:
  ROOT/<scene_id>/masks_agg/<frame_dir>.png

Aggregation mode:
  union (default): binary union (255 if any mask > threshold)
  count16: uint16 count map
  max: pixelwise max
  sum: sum of (mask>threshold) clipped to 255

Example:
  python batch_aggregate_from_txt.py \
    --root /nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv \
    --scene_ids_file /nas/qirui/only_in_train_val.txt \
    --mode union \
    --threshold 0
"""

import argparse
import re
from pathlib import Path
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_scene_ids(txt_path: Path):
    ids = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.search(r"\d+", s)
        if m:
            ids.append(m.group(0))
    return ids


def list_pngs_sorted(d: Path):
    return sorted(
        [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"],
        key=lambda p: natural_key(p.name)
    )


def load_gray(p: Path):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Failed to read: {p}")
    return m


def aggregate(mask_paths, mode, threshold):
    base = load_gray(mask_paths[0])
    H, W = base.shape[:2]

    if mode == "count16":
        acc = np.zeros((H, W), dtype=np.uint16)
        for p in mask_paths:
            m = load_gray(p)
            if m.shape[:2] != (H, W):
                raise ValueError(f"Size mismatch: {p}")
            acc[(m > threshold)] += 1
        return acc

    if mode == "max":
        acc = base.copy()
        for p in mask_paths[1:]:
            m = load_gray(p)
            if m.shape[:2] != (H, W):
                raise ValueError(f"Size mismatch: {p}")
            acc = np.maximum(acc, m)
        return acc.astype(np.uint8)

    if mode in ("union", "sum"):
        acc = np.zeros((H, W), dtype=np.uint16)
        for p in mask_paths:
            m = load_gray(p)
            if m.shape[:2] != (H, W):
                raise ValueError(f"Size mismatch: {p}")
            fg = (m > threshold).astype(np.uint16)
            if mode == "union":
                acc = np.maximum(acc, fg)
            else:
                acc += fg

        if mode == "union":
            return (acc > 0).astype(np.uint8) * 255
        return np.clip(acc * 255, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene_ids_file", required=True)
    ap.add_argument("--mode", default="union",
                    choices=["union", "sum", "count16", "max"])
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument("--clear_out", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")

    args = ap.parse_args()

    if cv2 is None:
        raise RuntimeError("cv2 not available")

    root = Path(args.root)
    scene_ids = read_scene_ids(Path(args.scene_ids_file))

    if not scene_ids:
        raise ValueError("scene_ids_file empty or invalid")

    total_frames_written = 0
    total_scenes = 0

    for sid in scene_ids:
        scene_dir = root / sid
        masks_dir = scene_dir / "masks"

        if not masks_dir.exists():
            print(f"[Skip] {sid} no masks dir")
            continue

        frame_dirs = sorted(
            [p for p in masks_dir.iterdir() if p.is_dir()],
            key=lambda p: natural_key(p.name)
        )

        if not frame_dirs:
            print(f"[Skip] {sid} no frame dirs")
            continue

        out_dir = scene_dir / "masks_agg"
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.clear_out:
            for p in out_dir.glob("*.png"):
                p.unlink()

        frames_written = 0

        for fd in frame_dirs:
            pngs = list_pngs_sorted(fd)
            if not pngs:
                continue

            out_png = out_dir / f"{fd.name}.png"

            if args.skip_existing and out_png.exists():
                continue

            agg = aggregate(pngs, args.mode, args.threshold)

            ok = cv2.imwrite(str(out_png), agg)
            if not ok:
                raise RuntimeError(f"Failed to write: {out_png}")

            frames_written += 1

        print(f"[Scene] {sid} frames_written={frames_written}")
        total_frames_written += frames_written
        total_scenes += 1

    print(f"[OK] scenes_done={total_scenes} total_frames_written={total_frames_written}")


if __name__ == "__main__":
    main()