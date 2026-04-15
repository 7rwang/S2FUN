#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-object / multi-GPU SAM3 video segmentation.

Design:
1. Read prompts from CSV with logic aligned to the image script:
   - contextual_object: normalize directly
   - interactive_objects: parse list/string, normalize to object-only prompt
2. Unlike the image script, we iterate over ALL interactive objects instead of only the first one.
3. Distributed execution uses prompt-level sharding:
   - each rank handles a subset of prompts
   - each prompt first searches initialization over candidate frames
   - after the first successful initialization, it is propagated through ALL frames
4. Output masks are written into a shared masks root:
   masks/0000/CTX__dresser__id000001.png
   masks/0000/INT__rotate_door_handle__id000123.png

Notes:
- For INT prompt "door handle", the actual video prompt is replaced with "rotate door handle".
- This version supports reading multiple returned masks per frame from the video predictor.
- Each returned object id inside one prompt/session is mapped to a globally unique instance id
  to avoid filename collisions across different prompts/sessions.
"""

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.visualization_utils import prepare_masks_for_visualization
except ImportError:
    print("Error: 找不到 sam3 库。")
    sys.exit(1)


# =============================
# distributed helpers
# =============================

def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


# =============================
# prompt parsing helpers
# aligned with image script, but iterate over ALL objects
# =============================

def normalize_prompt(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in {"none", "n/a", "na", ""}:
        return ""
    return re.sub(r"\s+", " ", s)


def strip_action_prefix_for_interactive_object(s: str) -> str:
    """
    Remove action prefixes from interactive object prompt.
    Examples:
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
    Rules aligned with image script:
    1) strip action prefix from interactive object prompt
    2) map "socket opening" -> "socket"
    3) if contains "drawer" but not "handle", force to "drawer handle"
    """
    s = normalize_prompt(x)
    if not s:
        return ""

    s = strip_action_prefix_for_interactive_object(s)
    low = s.strip().lower()

    if low == "socket opening":
        return "socket"

    if "drawer" in low and "handle" not in low:
        return "drawer handle"

    return normalize_prompt(s)


def maybe_replace_video_prompt(prompt: str, tag: str) -> str:
    """
    Video-specific replacement:
    - INT "door handle" -> "rotate door handle"
    """
    p = normalize_prompt(prompt)
    if tag == "INT" and p.lower() == "door handle":
        return "rotate door handle"
    return p


def dedup_keep_order(items):
    out = []
    seen = set()
    for x in items:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_interactive_objects_field(raw_value):
    """
    Difference from the image script:
    - image script only keeps the FIRST item
    - here we keep ALL objects and normalize each of them
    """
    raw = "" if raw_value is None else str(raw_value).strip()
    if not raw or raw.lower() in {"none", "n/a", "na"}:
        return []

    val = None
    try:
        val = ast.literal_eval(raw)
    except Exception:
        val = raw

    items = []
    if isinstance(val, (list, tuple)):
        items = [str(x) for x in val]
    else:
        # fallback: if plain string with commas, split; else single item
        if isinstance(val, str) and "," in val:
            items = [x.strip() for x in val.split(",") if x.strip()]
        else:
            items = [str(val)]

    normed = [normalize_interactive_object(x) for x in items]
    normed = [x for x in normed if x]
    return dedup_keep_order(normed)


def read_csv_prompts_multi(csv_path: Path):
    """
    Required columns:
      scene_id, contextual_object, interactive_objects

    Returns:
      scene2ctx: {scene_id: Counter(prompt -> count)}
      scene2int: {scene_id: Counter(prompt -> count)}
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

            all_int_objects = parse_interactive_objects_field(row.get("interactive_objects", ""))
            for obj in all_int_objects:
                scene2int[sid][obj] += 1

    return scene2ctx, scene2int


# =============================
# video / mask helpers
# =============================

def parse_init_frames(arg_str: str):
    frames = []
    if arg_str.strip():
        for x in arg_str.split(","):
            x = x.strip()
            if x:
                frames.append(int(x))
    return frames


def sanitize_class_name(name: str) -> str:
    name = (name or "").strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "unknown"


def load_jpeg_frames(frames_dir):
    frame_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))

    if not frame_paths:
        return [], []

    try:
        frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        print("  [警告] 帧文件名不是整数格式，使用字典序排序")
        frame_paths.sort()

    frames = []
    print(f"  [加载帧] 共找到 {len(frame_paths)} 个JPEG文件")
    for fpath in tqdm(frame_paths, desc="  读取帧"):
        frame = cv2.imread(fpath)
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print(f"  [警告] 无法读取: {fpath}")

    return frames, frame_paths


def save_mask_png(mask_item, save_path: Path):
    if isinstance(mask_item, torch.Tensor):
        m = mask_item.detach().cpu().numpy()
    else:
        m = np.asarray(mask_item)

    if m.ndim == 3:
        m = np.squeeze(m, 0)

    m = (m > 0).astype(np.uint8) * 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m, mode="L").save(save_path)


def mask_area(mask_item) -> int:
    if isinstance(mask_item, torch.Tensor):
        m = mask_item.detach().cpu().numpy()
    else:
        m = np.asarray(mask_item)
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    return int((m > 0).sum())


def prompt_has_mask(outputs: dict) -> bool:
    if not isinstance(outputs, dict):
        return False

    for k in ["out_binary_masks", "binary_masks", "masks", "out_masks"]:
        if k in outputs:
            m = outputs[k]
            try:
                if m is None:
                    continue
                if hasattr(m, "shape"):
                    if m.shape[0] > 0:
                        return True
                else:
                    if len(m) > 0:
                        return True
            except Exception:
                pass

    if "scores" in outputs:
        sc = outputs["scores"]
        try:
            if sc is not None and len(sc) > 0:
                return True
        except Exception:
            pass

    return False


def ensure_all_frame_dirs(mask_root: Path, num_frames: int):
    for fi in range(num_frames):
        (mask_root / f"{fi:04d}").mkdir(parents=True, exist_ok=True)


# =============================
# prompt construction / sharding
# =============================

def build_scene_prompts(csv_path: Path, scene_id: int):
    scene2ctx, scene2int = read_csv_prompts_multi(csv_path)

    ctx_prompts = [k for k, _ in scene2ctx.get(scene_id, Counter()).most_common()]
    int_prompts = [k for k, _ in scene2int.get(scene_id, Counter()).most_common()]

    all_prompts = []
    next_prompt_instance_id = 1

    for p in ctx_prompts:
        actual_prompt = maybe_replace_video_prompt(p, "CTX")
        all_prompts.append({
            "tag": "CTX",
            "source_prompt": p,
            "prompt": actual_prompt,
            "source_cls": sanitize_class_name(p),
            "cls": sanitize_class_name(actual_prompt),
            "instance_id": next_prompt_instance_id,
        })
        next_prompt_instance_id += 1

    for p in int_prompts:
        actual_prompt = maybe_replace_video_prompt(p, "INT")
        all_prompts.append({
            "tag": "INT",
            "source_prompt": p,
            "prompt": actual_prompt,
            "source_cls": sanitize_class_name(p),
            "cls": sanitize_class_name(actual_prompt),
            "instance_id": next_prompt_instance_id,
        })
        next_prompt_instance_id += 1

    for i, item in enumerate(all_prompts):
        item["global_prompt_idx"] = i

    return ctx_prompts, int_prompts, all_prompts


# =============================
# core processing
# =============================

def process_single_prompt_video(
    predictor,
    video_dir: str,
    prompt_item: dict,
    init_candidate_frames: list[int],
    min_mask_area: int,
):
    """
    One session per prompt.

    Scheme A:
      - try initialization frame by frame over the candidate frame list
      - stop at the FIRST frame that can produce a mask
      - then propagate through ALL frames

    Multi-mask version:
      - after propagation, iterate over all returned obj ids in each frame
      - map each returned obj id to a globally unique instance id
    """
    prompt_text = prompt_item["prompt"]
    prompt_seed_instance_id = int(prompt_item["instance_id"])

    resp = predictor.handle_request(dict(type="start_session", resource_path=video_dir))
    session_id = resp["session_id"]

    try:
        init_ok = False
        init_at = None

        for fidx in init_candidate_frames:
            r = predictor.handle_request(
                dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(fidx),
                    text=prompt_text,
                    obj_id=prompt_seed_instance_id,
                )
            )
            outs = r.get("outputs", {})
            if prompt_has_mask(outs):
                init_ok = True
                init_at = int(fidx)
                break

        if not init_ok:
            return {
                "init_ok": False,
                "init_at": None,
                "results": {},
                "returned_obj_to_global_instance": {},
            }

        results = {}
        for stream_resp in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            results[stream_resp["frame_index"]] = stream_resp["outputs"]

        fmt_masks = prepare_masks_for_visualization(results)

        kept_masks = defaultdict(list)  # {frame_idx: [(global_instance_id, returned_obj_id, mask), ...]}
        returned_obj_to_global_instance = {}
        next_local_offset = 0

        for frame_idx, masks_dict in fmt_masks.items():
            fi = int(frame_idx)
            md = masks_dict if isinstance(masks_dict, dict) else {}

            for returned_obj_id, m in md.items():
                if m is None:
                    continue
                if mask_area(m) < max(1, min_mask_area):
                    continue

                if returned_obj_id not in returned_obj_to_global_instance:
                    next_local_offset += 1
                    # Globally unique id across prompts/sessions.
                    # Keep the prompt seed instance id visible in the number layout.
                    global_instance_id = prompt_seed_instance_id * 1000 + next_local_offset
                    returned_obj_to_global_instance[returned_obj_id] = global_instance_id

                global_instance_id = returned_obj_to_global_instance[returned_obj_id]
                kept_masks[fi].append((global_instance_id, returned_obj_id, m))

        return {
            "init_ok": True,
            "init_at": init_at,
            "results": dict(kept_masks),
            "returned_obj_to_global_instance": returned_obj_to_global_instance,
        }

    finally:
        try:
            predictor.handle_request(dict(type="close_session", session_id=session_id))
        except Exception:
            pass


# =============================
# args
# =============================

def get_args():
    parser = argparse.ArgumentParser(
        description="SceneFun3D Video Segmentation with SAM3 - multi-object + multi-GPU"
    )
    parser.add_argument("--video-dir", type=str, required=True, help="包含JPEG帧序列的目录")
    parser.add_argument("--output-dir", type=str, required=True, help="结果输出目录")

    parser.add_argument("--csv-path", type=str, required=True,
                        help="CSV path. Will read contextual_object + interactive_objects from here.")
    parser.add_argument("--scene-id", type=int, required=True,
                        help="Scene id used to collect prompts from CSV.")

    parser.add_argument("--init-frames", type=str, default="",
                        help='显式指定初始化候选帧，例如 "0,30,60,90"；若留空，默认遍历整段视频的所有帧')
    parser.add_argument("--init-stride", type=int, default=1,
                        help="当未显式指定 --init-frames 时，候选初始化帧的步长。默认1，表示逐帧尝试")

    parser.add_argument("--save-masks", action="store_true",
                        help="保存每帧 mask PNG 到 <output_dir>/masks/xxxx/*.png")
    parser.add_argument("--masks-subdir", type=str, default="masks",
                        help="mask 保存子目录名")
    parser.add_argument("--force-frame-dirs", action="store_true",
                        help="即使某帧没有 mask，也创建对应空目录")
    parser.add_argument("--frame-count", type=int, default=-1,
                        help="强制指定总帧数；<=0 时使用实际帧数")

    parser.add_argument("--save-vis", action="store_true",
                        help="为每个 rank 输出一个 overlay 视频（仅叠加该 rank 负责的 prompts）")
    parser.add_argument("--output-fps", type=int, default=30,
                        help="输出视频帧率")

    parser.add_argument("--min-mask-area", type=int, default=1,
                        help="mask 面积小于该值则忽略")
    parser.add_argument("--dump-prompts-json", action="store_true",
                        help="保存 prompts_used.json")

    return parser.parse_args()


# =============================
# main
# =============================

def main():
    args = get_args()
    rank, world_size, local_rank = get_dist_info()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        predictor = build_sam3_video_predictor(
            checkpoint_path="/nas/qirui/sam3/checkpoints/sam3.pt",
            gpus_to_use=[local_rank],
        )
    else:
        predictor = build_sam3_video_predictor(
            checkpoint_path="/nas/qirui/sam3/checkpoints/sam3.pt",
            gpus_to_use=[],
        )

    print(f"\n[R{rank}] >>> 正在处理JPEG帧序列目录: {args.video_dir}")
    if not os.path.isdir(args.video_dir):
        print(f"[R{rank}] [错误] 目录不存在: {args.video_dir}")
        return

    vis_frames, frame_paths = load_jpeg_frames(args.video_dir)
    if len(vis_frames) == 0:
        print(f"[R{rank}] [错误] 没有找到可用的JPEG帧")
        return
    print(f"[R{rank}] [成功] 加载了 {len(vis_frames)} 帧")

    ctx_prompts, int_prompts, all_prompts = build_scene_prompts(Path(args.csv_path), int(args.scene_id))

    if len(all_prompts) == 0:
        print(f"[R{rank}] [错误] scene_id={args.scene_id} 没有读到任何 prompt")
        return

    if rank == 0:
        print(f"[R0] [prompts] CTX={len(ctx_prompts)} INT={len(int_prompts)} TOTAL={len(all_prompts)}")
        if args.dump_prompts_json:
            prompt_dump = {
                "scene_id": int(args.scene_id),
                "ctx_prompts": ctx_prompts,
                "int_prompts": int_prompts,
                "all_prompts": all_prompts,
                "world_size": world_size,
                "sharding": "prompt_idx % world_size == rank",
                "video_dir": args.video_dir,
            }
            out_json = Path(args.output_dir) / "prompts_used.json"
            out_json.write_text(json.dumps(prompt_dump, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[R0] [保存] prompts json: {out_json}")

    local_prompt_items = [p for p in all_prompts if (p["global_prompt_idx"] % world_size) == rank]
    print(f"[R{rank}] [shard] 本 rank 负责 {len(local_prompt_items)} / {len(all_prompts)} 个 prompts")

    init_candidate_frames = parse_init_frames(args.init_frames)
    if not init_candidate_frames:
        init_candidate_frames = list(range(0, len(vis_frames), max(args.init_stride, 1)))
    init_candidate_frames = [min(max(f, 0), len(vis_frames) - 1) for f in init_candidate_frames]
    init_candidate_frames = sorted(set(init_candidate_frames))
    preview = init_candidate_frames[:10]
    if len(init_candidate_frames) > 10:
        print(f"[R{rank}] [init] 将按顺序尝试 {len(init_candidate_frames)} 个候选帧，前10个: {preview} ...")
    else:
        print(f"[R{rank}] [init] 将按顺序尝试候选帧: {preview}")

    mask_root = Path(args.output_dir) / args.masks_subdir
    if args.save_masks:
        mask_root.mkdir(parents=True, exist_ok=True)
        num_frames = args.frame_count if args.frame_count and args.frame_count > 0 else len(vis_frames)
        print(f"[R{rank}] [mask] 输出根目录: {mask_root}")
        print(f"[R{rank}] [mask] 认为总帧数 num_frames={num_frames}")
        if args.force_frame_dirs and rank == 0:
            print(f"[R0] [mask] 强制创建所有帧目录...")
            ensure_all_frame_dirs(mask_root, num_frames)

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_nccl_available() and torch.cuda.is_available():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # ensure rank0 finishes frame-dir creation first
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    rank_overlay = defaultdict(list)  # frame_idx -> list[(mask, color)]
    prompt_summaries = []

    color_rng = np.random.RandomState(1000 + rank)
    instance_colors = {}

    try:
        pbar = tqdm(local_prompt_items, desc=f"[R{rank}] prompts", dynamic_ncols=True)
        for prompt_item in pbar:
            tag = prompt_item["tag"]
            prompt_text = prompt_item["prompt"]
            cls = prompt_item["cls"]
            gpidx = int(prompt_item["global_prompt_idx"])

            pbar.set_postfix(tag=tag, prompt=cls[:20])
            print(f"[R{rank}] [run] prompt_idx={gpidx} tag={tag} prompt={prompt_text!r}")

            res = process_single_prompt_video(
                predictor=predictor,
                video_dir=args.video_dir,
                prompt_item=prompt_item,
                init_candidate_frames=init_candidate_frames,
                min_mask_area=args.min_mask_area,
            )

            total_masks = sum(len(v) for v in res["results"].values())
            summary = {
                "global_prompt_idx": gpidx,
                "tag": tag,
                "source_prompt": prompt_item["source_prompt"],
                "prompt": prompt_text,
                "source_cls": prompt_item["source_cls"],
                "cls": cls,
                "prompt_seed_instance_id": int(prompt_item["instance_id"]),
                "init_ok": bool(res["init_ok"]),
                "init_at": res["init_at"],
                "num_frames_with_mask": len(res["results"]),
                "num_total_masks": total_masks,
                "returned_obj_to_global_instance": {
                    str(k): int(v) for k, v in res["returned_obj_to_global_instance"].items()
                },
                "rank": rank,
            }
            prompt_summaries.append(summary)

            if not res["init_ok"]:
                print(f"[R{rank}] [warn] 初始化失败: {prompt_text!r}")
                continue

            for fi, mask_items in res["results"].items():
                if args.save_masks:
                    frame_dir = mask_root / f"{int(fi):04d}"
                    frame_dir.mkdir(parents=True, exist_ok=True)

                    for global_instance_id, returned_obj_id, m in mask_items:
                        out_name = f"{tag}__{cls}__id{int(global_instance_id):06d}.png"
                        save_mask_png(m, frame_dir / out_name)

                if args.save_vis:
                    for global_instance_id, returned_obj_id, m in mask_items:
                        if global_instance_id not in instance_colors:
                            instance_colors[global_instance_id] = color_rng.randint(0, 255, size=(3,), dtype=np.uint8)
                        rank_overlay[int(fi)].append((m, instance_colors[global_instance_id]))

        # per-rank summary
        summary_path = Path(args.output_dir) / f"prompt_summary_rank{rank:02d}.json"
        summary_path.write_text(json.dumps(prompt_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[R{rank}] [保存] summary: {summary_path}")

        # optional per-rank visualization video
        if args.save_vis:
            h, w = vis_frames[0].shape[:2]
            video_basename = os.path.basename(args.video_dir.rstrip("/"))
            out_path = os.path.join(args.output_dir, f"{video_basename}_rank{rank:02d}_segmented.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (w, h))

            for fi, frame in enumerate(tqdm(vis_frames, desc=f"[R{rank}] 渲染视频", dynamic_ncols=True)):
                vis_frame = frame.copy()
                if fi in rank_overlay:
                    for m, color in rank_overlay[fi]:
                        vis_frame[m > 0] = (vis_frame[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                writer.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

            writer.release()
            print(f"[R{rank}] [保存] 可视化视频: {out_path}")

    finally:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            predictor.shutdown()
        except Exception:
            pass

    if rank == 0:
        print("\n[R0] >>> 处理完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[FATAL] rank={rank} local_rank={local_rank} exception={repr(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)
