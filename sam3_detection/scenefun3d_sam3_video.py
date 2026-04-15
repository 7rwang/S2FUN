#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA_VISIBLE_DEVICES=7 python scenefun3d_sam3_video.py \
  --video_dir /nas/qirui/scenefun3d/data/420683/42445137/hires_wide \
  --output_dir /nas/qirui/sam3/scenefun3d_ex/420683/video_outputs/anchor_cxt \
  --anchor-prompt "dresser" \
  --contextual-prompt "bottom drawer" \
  --init-frames "0" \
  --save-masks --force-frame-dirs
"""

import os
import re
import cv2
import torch
import argparse
import numpy as np
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.visualization_utils import prepare_masks_for_visualization
except ImportError:
    print("Error: 找不到 sam3 库。")
    exit(1)


# =============== helpers ===============

def parse_init_frames(arg_str: str):
    frames = []
    if arg_str.strip():
        for x in arg_str.split(","):
            x = x.strip()
            if x:
                frames.append(int(x))
    return frames


def sanitize_class_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
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
    """强制创建 masks/0000 ... masks/{num_frames-1:04d}"""
    for fi in range(num_frames):
        (mask_root / f"{fi:04d}").mkdir(parents=True, exist_ok=True)


# =============== args ===============

def get_args():
    parser = argparse.ArgumentParser(description="SceneFun3D Video Segmentation with SAM 3 (JPEG frames) - Anchor+Contextual")
    parser.add_argument("--video_dir", type=str, required=True, help="包含JPEG帧序列的目录")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")

    # anchor + contextual
    parser.add_argument("--anchor-prompt", type=str, required=True, help="Anchor object prompt（先初始化/先追踪）")
    parser.add_argument("--contextual-prompt", type=str, required=True, help="Contextual object prompt（后初始化/后追踪）")

    parser.add_argument("--init-frames", type=str, default="",
                        help='关键帧列表（帧号），例如 "0,30,60,90"；留空则用 init-stride 自动生成')
    parser.add_argument("--init-stride", type=int, default=30,
                        help="自动生成关键帧的步长（每隔多少帧尝试一次初始化），默认30")
    parser.add_argument("--max-init-tries", type=int, default=6,
                        help="每个物体最多尝试多少个关键帧，默认6（避免太慢）")

    parser.add_argument("--save-vis", action="store_true",
                        help="是否保存可视化视频（和你原脚本一致）")
    parser.add_argument("--merge-output", action="store_true",
                        help="是否合并所有批次的结果到一个视频（这里固定只有1批，保留开关不破坏用法）")
    parser.add_argument("--output-fps", type=int, default=30,
                        help="输出视频的帧率，默认30")

    # ===== 保存mask & 强制目录对齐 =====
    parser.add_argument("--save-masks", action="store_true",
                        help="保存每帧mask为PNG到 <output_dir>/masks/xxxx/*.png")
    parser.add_argument("--masks-subdir", type=str, default="masks",
                        help="mask保存子目录名，默认 masks")
    parser.add_argument("--force-frame-dirs", action="store_true",
                        help="即使某帧没有mask，也为每一帧创建空目录，保证与图像帧一一对应")
    parser.add_argument("--frame-count", type=int, default=-1,
                        help="强制指定总帧数（避免依赖vis_frames长度）。<=0 表示用 len(vis_frames)")

    # ===== 可选：最小面积阈值（防噪声碎片）=====
    parser.add_argument("--min-anchor-area", type=int, default=1,
                        help="anchor mask 面积小于该值则视为该帧无anchor（默认1，不过滤）")
    parser.add_argument("--min-ctx-area", type=int, default=1,
                        help="ctx mask 面积小于该值则不保存ctx（默认1，不过滤）")

    return parser.parse_args()


# =============== main ===============

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device_count = torch.cuda.device_count()
    predictor = build_sam3_video_predictor(
        checkpoint_path="/nas/qirui/sam3/checkpoints/sam3.pt",
        gpus_to_use=range(device_count),
    )

    print(f"\n>>> 正在处理JPEG帧序列目录: {args.video_dir}")
    if not os.path.isdir(args.video_dir):
        print(f"  [错误] 目录不存在: {args.video_dir}")
        return

    vis_frames, frame_paths = load_jpeg_frames(args.video_dir)
    if len(vis_frames) == 0:
        print("  [错误] 没有找到可用的JPEG帧")
        return
    print(f"  [成功] 加载了 {len(vis_frames)} 帧")

    anchor_prompt = args.anchor_prompt
    ctx_prompt = args.contextual_prompt
    print(f"  [prompts] anchor={anchor_prompt!r}  contextual={ctx_prompt!r}")
    print("  [coupling] 每帧：若无 anchor，则 ctx 不保存/不渲染（目录仍可空）")

    init_frames = parse_init_frames(args.init_frames)
    if not init_frames:
        init_frames = list(range(0, len(vis_frames), max(args.init_stride, 1)))
    init_frames = init_frames[: max(args.max_init_tries, 1)]
    init_frames = [min(max(f, 0), len(vis_frames) - 1) for f in init_frames]
    print(f"  [init] 将尝试关键帧: {init_frames}")

    h, w = vis_frames[0].shape[:2]
    video_basename = os.path.basename(args.video_dir.rstrip("/"))

    # ===== mask 根目录 & 强制创建空目录 =====
    mask_root = Path(args.output_dir) / args.masks_subdir
    if args.save_masks:
        mask_root.mkdir(parents=True, exist_ok=True)
        num_frames = args.frame_count if args.frame_count and args.frame_count > 0 else len(vis_frames)
        print(f"  [mask] 输出根目录: {mask_root}")
        print(f"  [mask] 认为总帧数 num_frames={num_frames}")

        if args.force_frame_dirs:
            print("  [mask] 强制创建所有帧目录（即使该帧无mask）...")
            ensure_all_frame_dirs(mask_root, num_frames)

    # ===== Start one session, add prompts in order: anchor -> ctx =====
    resp = predictor.handle_request(dict(type="start_session", resource_path=args.video_dir))
    session_id = resp["session_id"]

    # fixed obj ids
    ANCHOR_ID = 1
    CTX_ID = 2

    # init anchor first
    anchor_init_ok = False
    anchor_init_at = None
    try:
        for fidx in init_frames:
            r = predictor.handle_request(
                dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(fidx),
                    text=anchor_prompt,
                    obj_id=ANCHOR_ID,
                )
            )
            outs = r.get("outputs", {})
            if prompt_has_mask(outs):
                print(f"  - anchor 初始化成功: obj_id={ANCHOR_ID} @ frame={fidx}")
                anchor_init_ok = True
                anchor_init_at = fidx
                break
        if not anchor_init_ok:
            print("  [警告] anchor 在所有 init frames 上都初始化失败：将不会保存 ctx（因为耦合），也不会输出任何mask。")
            # 仍然可以输出空目录（force-frame-dirs 已做）
            predictor.handle_request(dict(type="close_session", session_id=session_id))
            predictor.shutdown()
            return

        # init ctx second (only if anchor ok)
        ctx_init_ok = False
        ctx_init_at = None
        for fidx in init_frames:
            r = predictor.handle_request(
                dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(fidx),
                    text=ctx_prompt,
                    obj_id=CTX_ID,
                )
            )
            outs = r.get("outputs", {})
            if prompt_has_mask(outs):
                print(f"  - ctx 初始化成功: obj_id={CTX_ID} @ frame={fidx}")
                ctx_init_ok = True
                ctx_init_at = fidx
                break
        if not ctx_init_ok:
            print("  [警告] ctx 初始化失败：后续将只保存/渲染 anchor（每帧最多1个anchor）。")

        # propagate
        print("  - 正在追踪 propagate_in_video ...")
        results = {}
        for stream_resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
            results[stream_resp["frame_index"]] = stream_resp["outputs"]

        fmt_masks = prepare_masks_for_visualization(results)
        # fmt_masks: {frame_idx: {obj_id: mask, ...}, ...}

        # ===== Per-frame coupling + save masks =====
        anchor_name = sanitize_class_name(anchor_prompt)
        ctx_name = sanitize_class_name(ctx_prompt)

        kept_masks = {}  # for rendering: {frame_idx: {obj_id: mask}}
        for frame_idx, masks_dict in fmt_masks.items():
            fi = int(frame_idx)
            md = masks_dict if isinstance(masks_dict, dict) else {}

            a_mask = md.get(ANCHOR_ID, None)
            c_mask = md.get(CTX_ID, None)

            # decide anchor existence per frame (area threshold)
            anchor_present = False
            if a_mask is not None and mask_area(a_mask) >= max(1, args.min_anchor_area):
                anchor_present = True

            frame_kept = {}
            if anchor_present:
                frame_kept[ANCHOR_ID] = a_mask

                # ctx is coupled: only consider saving if anchor exists
                if c_mask is not None and mask_area(c_mask) >= max(1, args.min_ctx_area):
                    frame_kept[CTX_ID] = c_mask
            # else: keep nothing (empty)

            if frame_kept:
                kept_masks[fi] = frame_kept

            # save pngs
            if args.save_masks:
                frame_dir = mask_root / f"{fi:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)  # on-demand even without force-frame-dirs

                if anchor_present:
                    save_mask_png(a_mask, frame_dir / f"ANCHOR__{anchor_name}.png")
                    if CTX_ID in frame_kept:
                        save_mask_png(c_mask, frame_dir / f"CTX__{ctx_name}.png")
                # else: save nothing (keep empty dir for alignment)

        if args.save_masks:
            print("  [mask] 保存完成：每帧最多 ANCHOR__*.png + CTX__*.png；且无anchor帧为空目录（如启用force）")

        # ===== Render video (single batch) =====
        if args.save_vis or args.merge_output or (not args.merge_output):
            out_path = os.path.join(args.output_dir, f"{video_basename}_anchor_ctx_segmented.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (w, h))

            # deterministic colors
            # anchor: red-ish, ctx: green-ish (但不写死颜色值，用随机但固定seed)
            np.random.seed(42)
            colors = np.random.randint(0, 255, (3, 3), dtype=np.uint8)  # idx 1,2 used

            for fi, frame in enumerate(tqdm(vis_frames, desc="  渲染可视化视频")):
                vis_frame = frame.copy()
                if fi in kept_masks:
                    md = kept_masks[fi]
                    # anchor
                    if ANCHOR_ID in md:
                        color = colors[ANCHOR_ID].tolist()
                        m = md[ANCHOR_ID]
                        vis_frame[m > 0] = (vis_frame[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    # ctx
                    if CTX_ID in md:
                        color = colors[CTX_ID].tolist()
                        m = md[CTX_ID]
                        vis_frame[m > 0] = (vis_frame[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

                writer.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

            writer.release()
            print(f"  [保存] 可视化视频: {out_path}")

    finally:
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        torch.cuda.empty_cache()
        predictor.shutdown()

    print("\n>>> 处理完成！")


if __name__ == "__main__":
    main()
