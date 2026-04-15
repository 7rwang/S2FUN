#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python count_masks_per_frame_jsonl.py \
  --root /nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv \
  --scene_ids_file /nas/qirui/only_in_train_val.txt \
  --out_jsonl /nas/qirui/mask_stats.jsonl
"""
import argparse
import json
import re
from pathlib import Path


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


def list_pngs(d: Path):
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="anchor_cxt_csv root")
    ap.add_argument("--scene_ids_file", required=True,
                    help="txt containing scene ids")
    ap.add_argument("--out_jsonl", required=True,
                    help="output jsonl file")
    ap.add_argument("--simple_format", action="store_true",
                    help="If set, output 'frame:count' string instead of full json object")

    args = ap.parse_args()

    root = Path(args.root)
    scene_ids = read_scene_ids(Path(args.scene_ids_file))

    if not scene_ids:
        raise ValueError("scene_ids_file empty or invalid")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0

    with out_path.open("w", encoding="utf-8") as f_out:
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

            for fd in frame_dirs:
                pngs = list_pngs(fd)
                count = len(pngs)

                if args.simple_format:
                    # 输出类似： 0001:15
                    line = f"{fd.name}:{count}"
                else:
                    obj = {
                        "scene_id": sid,
                        "frame": fd.name,
                        "mask_count": count
                    }
                    line = json.dumps(obj, ensure_ascii=False)

                f_out.write(line + "\n")
                total_frames += 1

    print(f"[OK] wrote {total_frames} frames to {out_path}")


if __name__ == "__main__":
    main()