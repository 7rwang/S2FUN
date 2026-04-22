#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Examples
--------
1) Batch scene mode (original behavior)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 scenefun3d_sam3_image_dual_mode.py \
  --csv-path /nas/qirui/sam3/scenefun3d_ex/src/parse_result_30scenes.csv \
  --data-root /nas/qirui/scenefun3d/val \
  --output-root /nas/qirui/sam3/scenefun3d_ex/experiments/exp_01_doorhandle_fallback \
  --image-subdir hires_wide \
  --thr 0.5 \
  --save-vis \
  --scene-id 466162

2) Single image mode with manual prompts
python scenefun3d_sam3_image_dual_mode.py \
  --image-path /path/to/image.jpg \
  --output-root /tmp/sam3_single \
  --ctx-prompts cabinet door drawer \
  --int-prompts "door handle" "light switch" \
  --thr 0.5 \
  --save-vis

3) Single image mode using prompts from json
python scenefun3d_sam3_image_dual_mode.py \
  --image-path /path/to/image.jpg \
  --output-root /tmp/sam3_single \
  --prompts-json /path/to/prompts.json \
  --thr 0.5

prompts.json format:
{
  "ctx_prompts": ["cabinet", "drawer"],
  "int_prompts": ["door handle", "light switch"]
}
"""

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
import traceback
import yaml
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


_CMAP = plt.get_cmap("tab20")
_COLORS = [_CMAP(i) for i in range(20)]


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def natural_key(p: Path):
    parts = re.split(r"(\d+)", p.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def sanitize_class_name(name: str) -> str:
    name = (name or "").strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "unknown"


def normalize_prompt(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in {"none", "n/a", "na", ""}:
        return ""
    return re.sub(r"\s+", " ", s)


def normalize_prompt_list(values):
    if values is None:
        return []

    out = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and "," in v:
            pieces = [x.strip() for x in v.split(",")]
        else:
            pieces = [v]

        for p in pieces:
            p = normalize_prompt(p)
            if p:
                out.append(p)

    deduped = []
    seen = set()
    for x in out:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)
    return deduped


def strip_action_prefix_for_interactive_object(s: str) -> str:
    """
    Remove action prefixes from interactive object prompt.
    Example:
        "hook_pull door handle" -> "door handle"
        "rotate door handle" -> "door handle"
        "push button" -> "button"
    """
    s = normalize_prompt(s)
    if not s:
        return ""

    low = s.lower().strip()

    action_prefixes = [
        "hook_pull ",
        "hook pull ",
        "rotate ",
        "pull ",
        "push ",
        "twist ",
        "turn ",
        "press ",
        "pinch_pull ",
        "pinch pull ",
        "key_press ",
        "key press ",
        "unplug ",
    ]

    for prefix in action_prefixes:
        if low.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    return normalize_prompt(s)


def normalize_interactive_object(x: str) -> str:
    """
    Rules:
    1) Only keep the FIRST item if there are multiple (comma-separated).
       e.g. "drawer handle,drawer front" -> "drawer handle"
    2) Remove action prefix from interactive object prompt.
       e.g. "hook_pull door handle" -> "door handle"
    3) If it is "socket opening" (case-insensitive), map to "socket".
    4) If it contains "drawer":
         - if already contains "handle": keep as-is
         - else: force to "drawer handle"
    """
    s = normalize_prompt(x)
    if not s:
        return ""

    if "," in s:
        s = s.split(",", 1)[0].strip()

    s = strip_action_prefix_for_interactive_object(s)
    low = s.strip().lower()

    if low == "socket opening":
        return "socket"

    if "drawer" in low and "handle" not in low:
        return "drawer handle"

    return normalize_prompt(s)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mask_area(mask, thr=0.5):
    m = _to_numpy(mask)
    if m is None:
        return 0
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    return int((m > thr).sum())


def binarize_mask(mask, thr=0.5):
    m = _to_numpy(mask)
    if m is None:
        return None
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    return (m > thr).astype(np.uint8)


def mask_iou(mask_a, mask_b, thr=0.5):
    a = binarize_mask(mask_a, thr)
    b = binarize_mask(mask_b, thr)
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_overlap_on_smaller(mask_a, mask_b, thr=0.5):
    """
    overlap ratio measured on the smaller mask:
        inter / min(area_a, area_b)
    """
    a = binarize_mask(mask_a, thr)
    b = binarize_mask(mask_b, thr)
    if a is None or b is None:
        return 0.0

    inter = np.logical_and(a, b).sum()
    area_a = a.sum()
    area_b = b.sum()
    small = min(area_a, area_b)

    if small == 0:
        return 0.0
    return float(inter) / float(small)


def dedup_int_masks_keep_smaller(
    masks,
    boxes,
    thr=0.5,
    overlap_thr=0.9,
):
    """
    Only for INT prompts:
    If two masks have very high overlap w.r.t. the smaller one,
    keep the smaller-area one.
    """
    if masks is None or len(masks) <= 1:
        return masks, boxes

    n = len(masks)
    keep = [True] * n
    areas = [mask_area(m, thr) for m in masks]

    for i in range(n):
        if not keep[i] or areas[i] <= 0:
            continue
        for j in range(i + 1, n):
            if not keep[j] or areas[j] <= 0:
                continue

            overlap_small = mask_overlap_on_smaller(masks[i], masks[j], thr)

            if overlap_small >= overlap_thr:
                if areas[i] <= areas[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    new_masks = [masks[i] for i in range(n) if keep[i]]
    new_boxes = None
    if boxes is not None:
        new_boxes = [boxes[i] for i in range(min(n, len(boxes))) if keep[i]]

    return new_masks, new_boxes


def dedup_frame_level_door_handles(
    items,
    thr=0.5,
    overlap_thr=0.7,
):
    """
    Frame-level dedup for all door_handle masks in one frame.

    items: list of dict, each item contains:
        {
            "tag": "INT",
            "cls": "door_handle",
            "mask": ...,
            "box": ...,
            "prompt": ...,
            "area": ...
        }

    Rule:
    - only compare items whose cls == "door_handle"
    - if overlap on smaller >= overlap_thr, keep the smaller-area one
    """
    if items is None or len(items) <= 1:
        return items

    n = len(items)
    keep = [True] * n
    areas = [mask_area(it["mask"], thr) for it in items]

    for i in range(n):
        if not keep[i] or areas[i] <= 0:
            continue
        for j in range(i + 1, n):
            if not keep[j] or areas[j] <= 0:
                continue

            overlap_small = mask_overlap_on_smaller(items[i]["mask"], items[j]["mask"], thr)

            if overlap_small >= overlap_thr:
                prompt_i = items[i].get('prompt', '').lower()
                prompt_j = items[j].get('prompt', '').lower()

                if (prompt_i == prompt_j) or (prompt_i not in ['door handle', 'rotate door handle'] and prompt_j not in ['door handle', 'rotate door handle']):
                    if areas[i] <= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                elif prompt_i == 'rotate door handle' and prompt_j == 'door handle':
                    if areas[i] < areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                elif prompt_i == 'door handle' and prompt_j == 'rotate door handle':
                    if areas[j] < areas[i]:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False
                else:
                    if areas[i] <= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

    return [items[i] for i in range(n) if keep[i]]


def save_mask_png(mask, path: Path, thr=0.5):
    m = _to_numpy(mask)
    if m is None:
        return
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    m = (m > thr).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m, mode="L").save(path)


def set_image_get_state(processor: Sam3Processor, img: Image.Image):
    st = processor.set_image(img)
    if st is not None:
        return st

    for attr in ("state", "_state", "inference_state", "_inference_state", "image_state", "_image_state"):
        if hasattr(processor, attr):
            v = getattr(processor, attr)
            if v is not None:
                return v

    raise ValueError(
        "You must call set_image before set_text_prompt, but set_image returned None "
        "and no internal state attribute was found."
    )


def save_vis(img_pil: Image.Image, overlays: list[dict], out_path: Path, thr=0.5, title: str | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_pil)
    ax.axis("off")
    if title:
        ax.set_title(title)

    for i, ov in enumerate(overlays):
        color = _COLORS[i % len(_COLORS)]
        mask = _to_numpy(ov.get("mask"))
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = np.squeeze(mask, 0)

        rgba = np.array([color[0], color[1], color[2], 0.45], dtype=np.float32)
        ax.imshow((mask > thr)[..., None] * rgba)

        box = ov.get("box", None)
        if box is not None:
            b = _to_numpy(box)
            if b is not None and len(b) == 4:
                if hasattr(b, "tolist"):
                    b = b.tolist()
                x0, y0, x1, y1 = map(float, b)
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=1.5,
                        edgecolor=(color[0], color[1], color[2], 1.0),
                        facecolor="none",
                    )
                )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_csv_prompts(csv_path: Path):
    """
    Required columns:
      scene_id, contextual_object, interactive_objects

    interactive_objects processing:
    - ONLY keep the first item
    - strip action prefix
    - normalize to object-only prompt
    """
    scene2ctx = defaultdict(Counter)
    scene2int = defaultdict(Counter)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "contextual_object", "interactive_objects"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            sid = (row.get("scene_id") or "").strip()
            if not sid.isdigit():
                continue
            sid = int(sid)

            ctx = normalize_prompt(row.get("contextual_object"))
            if ctx:
                scene2ctx[sid][ctx] += 1

            raw = row.get("interactive_objects", "")
            raw = "" if raw is None else str(raw).strip()

            val = None
            if raw and raw.lower() not in {"none", "n/a", "na"}:
                try:
                    val = ast.literal_eval(raw)
                except Exception:
                    val = raw

            first_item = ""
            if isinstance(val, (list, tuple)):
                if len(val) > 0:
                    first_item = str(val[0])
            elif val is not None:
                first_item = str(val)

            first_item = normalize_interactive_object(first_item)
            if first_item:
                scene2int[sid][first_item] += 1

    return scene2ctx, scene2int


def find_first_sequence_image_dir(data_root: Path, scene_id: int, image_subdir: str):
    scene_dir = data_root / str(scene_id)
    if not scene_dir.exists():
        return None

    sequences = sorted([p for p in scene_dir.iterdir() if p.is_dir()], key=natural_key)
    if not sequences:
        return None

    first_seq = sequences[0]
    img_dir = first_seq / image_subdir
    if not img_dir.exists() or not img_dir.is_dir():
        return None
    return img_dir


def setup_logging(log_file=None, level=logging.INFO):
    if log_file is None:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"sam3_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def create_experiment_config(mode, world_size, rank, cuda_available, cuda_device_count,
                             current_device, threshold=0.5, **kwargs):
    config = {
        'experiment': {
            'timestamp': datetime.now().isoformat(),
            'script_name': 'scenefun3d_sam3_image_dual_mode.py',
            'version': '2.0.0',
            'mode': mode,
            'description': 'Support both batch-scene mode and single-image mode'
        },
        'hardware': {
            'world_size': world_size,
            'rank': rank,
            'cuda_available': cuda_available,
            'cuda_device_count': cuda_device_count,
            'current_device': current_device
        },
        'model': {
            'name': 'SAM3',
            'threshold': threshold,
            'save_visualization': kwargs.get('save_vis', False)
        },
        'features': {
            'read_int_prompt_object_only': True,
            'door_handle_secondary_segmentation': True,
            'door_handle_secondary_prompt_fixed': 'rotate door handle',
            'secondary_segmentation_criteria': 'smaller_area_wins',
            'int_duplicate_suppression': True,
            'int_duplicate_metric': 'intersection_over_smaller_area',
            'int_duplicate_overlap_threshold': 0.9,
            'int_duplicate_keep_rule': 'keep_smaller_area',
            'frame_level_door_handle_dedup': True,
            'frame_level_door_handle_metric': 'intersection_over_smaller_area',
            'frame_level_door_handle_overlap_threshold': 0.7,
            'frame_level_door_handle_keep_rule': 'keep_smaller_area'
        },
        'data': kwargs,
    }
    return config


def save_experiment_config(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, f"experiment_config_{config['experiment']['timestamp'].replace(':', '-')}.yaml")

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Experiment configuration saved to: {config_file}")
    return config_file


def is_door_handle_prompt(prompt: str) -> bool:
    if not prompt:
        return False
    prompt_lower = prompt.lower().strip()
    return prompt_lower == "door handle"


def get_secondary_door_handle_prompt(original_prompt: str) -> str:
    if is_door_handle_prompt(original_prompt):
        return "rotate door handle"
    return None


def perform_secondary_segmentation(processor, state, original_prompt: str, original_masks: list, original_boxes: list, thr: float):
    secondary_prompt = get_secondary_door_handle_prompt(original_prompt)

    if secondary_prompt is None:
        return original_masks, original_boxes, original_prompt

    try:
        secondary_out = processor.set_text_prompt(state=state, prompt=secondary_prompt)
        secondary_masks = secondary_out.get("masks", [])
        secondary_boxes = secondary_out.get("boxes", [])

        if secondary_masks is None or len(secondary_masks) == 0:
            return original_masks, original_boxes, original_prompt

        original_total_area = sum(mask_area(m, thr) for m in original_masks)
        secondary_total_area = sum(mask_area(m, thr) for m in secondary_masks)

        if secondary_total_area > 0 and secondary_total_area < original_total_area:
            return secondary_masks, secondary_boxes, secondary_prompt
        else:
            return original_masks, original_boxes, original_prompt

    except Exception:
        return original_masks, original_boxes, original_prompt


def list_images(d: Path):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(list(d.glob(ext)))
    return sorted(imgs, key=natural_key)


def load_prompts_from_json(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ctx_prompts = normalize_prompt_list(data.get("ctx_prompts", []))
    int_prompts = normalize_prompt_list(data.get("int_prompts", []))
    return ctx_prompts, int_prompts


def resolve_prompts(args):
    ctx_prompts = []
    int_prompts = []
    prompt_source = None

    if args.prompts_json is not None:
        ctx_prompts, int_prompts = load_prompts_from_json(Path(args.prompts_json))
        prompt_source = f"prompts_json:{args.prompts_json}"

    manual_ctx = normalize_prompt_list(args.ctx_prompts)
    manual_int = normalize_prompt_list(args.int_prompts)

    if manual_ctx or manual_int:
        ctx_prompts = manual_ctx
        int_prompts = manual_int
        prompt_source = "manual_args"

    if (not ctx_prompts and not int_prompts) and args.csv_path is not None and args.scene_id is not None:
        scene2ctx, scene2int = read_csv_prompts(Path(args.csv_path))
        scene_id = int(args.scene_id)
        ctx_prompts = [k for k, _ in scene2ctx.get(scene_id, Counter()).most_common()]
        int_prompts = [k for k, _ in scene2int.get(scene_id, Counter()).most_common()]
        prompt_source = f"csv_scene:{args.scene_id}"

    return ctx_prompts, int_prompts, prompt_source


def process_single_frame(
    processor: Sam3Processor,
    img: Image.Image,
    all_prompts,
    frame_dir: Path,
    vis_out: Path | None,
    thr: float,
    vis_title: str | None = None,
):
    state = set_image_get_state(processor, img)
    frame_dir.mkdir(parents=True, exist_ok=True)

    overlays = []
    frame_results = []

    for tag, prompt in all_prompts:
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks = out.get("masks", [])
        boxes = out.get("boxes", [])

        if masks is None or len(masks) == 0:
            continue

        final_masks, final_boxes, final_prompt = perform_secondary_segmentation(
            processor, state, prompt, masks, boxes, thr
        )

        if final_prompt.lower() == "door handle" and len(final_masks) > 1:
            areas = [mask_area(m, thr) for m in final_masks]
            valid_areas = [a for a in areas if a > 0]
            if valid_areas:
                min_area = min(valid_areas)
                best_idx = areas.index(min_area)
                final_masks = [final_masks[best_idx]]
                if final_boxes is not None and len(final_boxes) > best_idx:
                    final_boxes = [final_boxes[best_idx]]
                else:
                    final_boxes = None

        cls = sanitize_class_name(final_prompt)

        if tag == "INT":
            final_masks, final_boxes = dedup_int_masks_keep_smaller(
                final_masks,
                final_boxes,
                thr=thr,
                overlap_thr=0.9,
            )

        for mi, m in enumerate(final_masks):
            area = mask_area(m, thr)
            if area <= 0:
                continue

            box = final_boxes[mi] if final_boxes is not None and mi < len(final_boxes) else None

            frame_results.append({
                "tag": tag,
                "cls": cls,
                "mask": m,
                "box": box,
                "area": area,
                "prompt": final_prompt,
            })

    door_items = []
    other_items = []
    for item in frame_results:
        if item["tag"] == "INT" and item["cls"] == "door_handle":
            door_items.append(item)
        else:
            other_items.append(item)

    if len(door_items) > 1:
        door_items = dedup_frame_level_door_handles(
            door_items,
            thr=thr,
            overlap_thr=0.7,
        )

    frame_results = other_items + door_items

    class_counters = defaultdict(int)
    for item in frame_results:
        tag = item["tag"]
        cls = item["cls"]
        m = item["mask"]
        box = item["box"]
        area = item["area"]

        idx = class_counters[(tag, cls)]
        class_counters[(tag, cls)] += 1

        save_mask_png(
            m,
            frame_dir / f"{tag}__{cls}__{idx:03d}__area{area}.png",
            thr,
        )
        overlays.append({"mask": m, "box": box})

    if vis_out is not None:
        save_vis(img_pil=img, overlays=overlays, out_path=vis_out, thr=thr, title=vis_title)

    summary = {
        "num_results": len(frame_results),
        "results": [
            {
                "tag": item["tag"],
                "cls": item["cls"],
                "area": int(item["area"]),
                "prompt": item["prompt"],
            }
            for item in frame_results
        ]
    }
    return summary


def run_scene_frame_sharded(
    processor: Sam3Processor,
    scene_id: int,
    image_dir: Path,
    ctx_prompts: list[str],
    int_prompts: list[str],
    out_root: Path,
    thr: float,
    save_vis_flag: bool,
    rank: int,
    world_size: int,
):
    scene_out = out_root / str(scene_id)
    masks_root = scene_out / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    num_frames = len(images)

    all_prompts = [("CTX", p) for p in ctx_prompts] + [("INT", p) for p in int_prompts]

    local_indices = [i for i in range(num_frames) if (i % world_size) == rank]
    local_total = len(local_indices)

    if rank == 0:
        logging.info(f"Starting frame processing for scene {scene_id}")
        logging.info(f"Total frames: {num_frames}, Local frames: {local_total}")
        logging.info(f"Total prompts: {len(all_prompts)} (CTX: {len(ctx_prompts)}, INT: {len(int_prompts)})")

    pbar = tqdm(
        local_indices,
        total=local_total,
        desc=f"[R{rank}] frames",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for fi in pbar:
        img_path = images[fi]
        img = Image.open(img_path).convert("RGB")

        frame_dir = masks_root / str(fi)
        vis_out = scene_out / "vis" / f"{fi:06d}.png" if save_vis_flag else None

        process_single_frame(
            processor=processor,
            img=img,
            all_prompts=all_prompts,
            frame_dir=frame_dir,
            vis_out=vis_out,
            thr=thr,
            vis_title=f"{scene_id} {image_dir.parent.name} frame={fi:06d} rank={rank}",
        )

    if rank == 0:
        logging.info(f"Frame processing completed for scene {scene_id}. Processed {local_total} frames on rank {rank}.")


def run_single_image(
    processor: Sam3Processor,
    image_path: Path,
    ctx_prompts: list[str],
    int_prompts: list[str],
    out_root: Path,
    thr: float,
    save_vis_flag: bool,
    rank: int,
):
    if rank != 0:
        logging.info("Single-image mode only runs on rank 0. Other ranks exit without processing.")
        return

    img = Image.open(image_path).convert("RGB")
    stem = sanitize_class_name(image_path.stem)
    single_out = out_root / "single_image" / stem
    masks_root = single_out / "masks"
    summary_path = single_out / "summary.json"
    vis_out = single_out / "vis.png" if save_vis_flag else None

    all_prompts = [("CTX", p) for p in ctx_prompts] + [("INT", p) for p in int_prompts]
    summary = process_single_frame(
        processor=processor,
        img=img,
        all_prompts=all_prompts,
        frame_dir=masks_root,
        vis_out=vis_out,
        thr=thr,
        vis_title=f"single_image: {image_path.name}",
    )
    summary.update({
        "image_path": str(image_path),
        "ctx_prompts": ctx_prompts,
        "int_prompts": int_prompts,
    })
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Single-image processing completed: {image_path}")
    logging.info(f"Summary saved to: {summary_path}")


def validate_args(args):
    has_scene_mode = args.scene_id is not None
    has_single_image_mode = args.image_path is not None

    if has_scene_mode and has_single_image_mode:
        raise ValueError("Choose only one mode: either --scene-id (batch scene mode) or --image-path (single image mode).")

    if not has_scene_mode and not has_single_image_mode:
        raise ValueError("You must provide either --scene-id for batch scene mode or --image-path for single image mode.")

    if has_scene_mode:
        if args.csv_path is None:
            raise ValueError("Batch scene mode requires --csv-path.")
        if args.data_root is None:
            raise ValueError("Batch scene mode requires --data-root.")

    if has_single_image_mode:
        if args.prompts_json is None and not args.ctx_prompts and not args.int_prompts:
            if not (args.csv_path is not None and args.scene_id is not None):
                raise ValueError(
                    "Single image mode requires prompts. Provide one of: --prompts-json, --ctx-prompts/--int-prompts."
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--scene-id", type=int, default=None, help="Run ONE scene; all GPUs collaborate on it.")
    ap.add_argument("--image-path", default=None, help="Run a single image.")
    ap.add_argument("--prompts-json", default=None, help="JSON file with keys: ctx_prompts, int_prompts")
    ap.add_argument("--ctx-prompts", nargs="*", default=None, help="Manual contextual prompts. Supports space-separated or comma-separated values.")
    ap.add_argument("--int-prompts", nargs="*", default=None, help="Manual interactive prompts. Supports space-separated or comma-separated values.")
    ap.add_argument("--image-subdir", default="hires_wide")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save-vis", action="store_true")
    args = ap.parse_args()

    validate_args(args)

    rank, world_size, local_rank = get_dist_info()

    if rank == 0:
        setup_logging()
        logging.info("Starting SAM3 detection experiment")
        logging.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if rank == 0:
            logging.info(
                f"CUDA available: {torch.cuda.is_available()}, "
                f"device count: {torch.cuda.device_count()}, current device: {local_rank}"
            )

    ctx_prompts, int_prompts, prompt_source = resolve_prompts(args)
    if not ctx_prompts and not int_prompts:
        raise RuntimeError("No prompts resolved. Check your CSV / prompts-json / manual prompt arguments.")

    out_root = Path(args.output_root)

    model = build_sam3_image_model()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    mode = "single_image" if args.image_path is not None else "batch_scene"

    if rank == 0:
        config_kwargs = {
            'output_root': str(out_root),
            'prompt_source': prompt_source,
            'ctx_prompts_count': len(ctx_prompts),
            'int_prompts_count': len(int_prompts),
            'ctx_prompts': ctx_prompts,
            'int_prompts': int_prompts,
            'save_vis': args.save_vis,
        }
        if mode == "batch_scene":
            data_root = Path(args.data_root)
            scene_out = out_root / str(args.scene_id)
            scene_out.mkdir(parents=True, exist_ok=True)
            config_kwargs.update({
                'scene_id': int(args.scene_id),
                'csv_path': args.csv_path,
                'data_root': str(data_root),
                'image_subdir': args.image_subdir,
                'sharding_strategy': 'frame_idx % world_size == rank',
                'frame_distribution': f'rank {rank} of {world_size}',
            })
            prompt_dump = {
                "scene_id": int(args.scene_id),
                "image_subdir": args.image_subdir,
                "ctx_prompts": ctx_prompts,
                "int_prompts": int_prompts,
                "world_size": world_size,
                "sharding": "frame_idx % world_size == rank",
                "prompt_source": prompt_source,
            }
            (scene_out / "prompts_used.json").write_text(
                json.dumps(prompt_dump, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            save_experiment_config(
                create_experiment_config(
                    mode=mode,
                    world_size=world_size,
                    rank=rank,
                    cuda_available=torch.cuda.is_available(),
                    cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    current_device=local_rank,
                    threshold=args.thr,
                    **config_kwargs,
                ),
                str(scene_out),
            )
        else:
            single_root = out_root / "single_image" / sanitize_class_name(Path(args.image_path).stem)
            single_root.mkdir(parents=True, exist_ok=True)
            config_kwargs.update({
                'image_path': args.image_path,
            })
            save_experiment_config(
                create_experiment_config(
                    mode=mode,
                    world_size=world_size,
                    rank=rank,
                    cuda_available=torch.cuda.is_available(),
                    cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    current_device=local_rank,
                    threshold=args.thr,
                    **config_kwargs,
                ),
                str(single_root),
            )

    with torch.no_grad():
        if mode == "batch_scene":
            data_root = Path(args.data_root)
            scene_id = int(args.scene_id)
            img_dir = find_first_sequence_image_dir(data_root, scene_id, args.image_subdir)
            if img_dir is None:
                raise RuntimeError(
                    f"scene {scene_id}: cannot find first sequence with subdir '{args.image_subdir}' under {data_root}/{scene_id}"
                )
            if rank == 0:
                scene_out = out_root / str(scene_id)
                prompt_dump_path = scene_out / "prompts_used.json"
                if prompt_dump_path.exists():
                    prompt_dump = json.loads(prompt_dump_path.read_text(encoding="utf-8"))
                    prompt_dump["sequence_used"] = img_dir.parent.name
                    prompt_dump["image_subdir"] = img_dir.name
                    prompt_dump_path.write_text(json.dumps(prompt_dump, indent=2, ensure_ascii=False), encoding="utf-8")
            run_scene_frame_sharded(
                processor=processor,
                scene_id=scene_id,
                image_dir=img_dir,
                ctx_prompts=ctx_prompts,
                int_prompts=int_prompts,
                out_root=out_root,
                thr=args.thr,
                save_vis_flag=args.save_vis,
                rank=rank,
                world_size=world_size,
            )
        else:
            run_single_image(
                processor=processor,
                image_path=Path(args.image_path),
                ctx_prompts=ctx_prompts,
                int_prompts=int_prompts,
                out_root=out_root,
                thr=args.thr,
                save_vis_flag=args.save_vis,
                rank=rank,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[FATAL] rank={rank} local_rank={local_rank} exception={repr(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)
