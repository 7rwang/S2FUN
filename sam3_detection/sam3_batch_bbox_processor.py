#!/usr/bin/env python3
"""
批量处理 bbox prompt 的 SAM3 脚本

功能：
1. 从 JSON 读取每帧 bbox
2. 对每帧的多个 bbox 分别跑 SAM3
3. 按 scene_id 输出目录
4. 只保存 mask，不保存原图
5. 保存每一帧所有 mask 到 masks/<frame_id>/
6. 即使某一帧没有 bbox 或没有 mask，也会创建空的 masks/<frame_id>/ 目录
7. 可选保存该帧所有 mask 的 merge 并集图 merged_mask.png

支持两种 bbox JSON 格式：

1) 每帧是 list[list[4]]
{
  "30": [
    [30, 0, 86, 179],
    [100, 120, 180, 240]
  ]
}

2) 每帧是 dict[str, list[4]]
{
  "30": {
    "Close the bedroom door": [30, 0, 86, 179],
    "Open the first drawer": [100, 120, 180, 240]
  }
}

Usage:
CUDA_VISIBLE_DEVICES=1 python sam3_batch_bbox_processor.py \
  --bbox-json /nas/qirui/sam3/S2FUN/bbox.json \
  --scene-dir /nas/qirui/scenefun3d/val/421254 \
  --output-dir /nas/qirui/sam3/scenefun3d_ex/experiments4bboxprompt \
  --save-merged-mask
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def natural_key(p):
    parts = re.split(r"(\d+)", str(p))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def find_first_sequence_image_dir(scene_dir, image_subdir="hires_wide"):
    """
    找到场景的第一个可用图像目录。

    支持两种目录结构：
    1) scene_dir/hires_wide
    2) scene_dir/0031/hires_wide
    """
    scene_path = Path(scene_dir)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_path}")

    direct_img_dir = scene_path / image_subdir
    if direct_img_dir.exists() and direct_img_dir.is_dir():
        return direct_img_dir

    subdirs = [p for p in scene_path.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {scene_path}")

    subdirs.sort(key=natural_key)

    for seq_dir in subdirs:
        img_dir = seq_dir / image_subdir
        if img_dir.exists() and img_dir.is_dir():
            return img_dir

    raise RuntimeError(
        f"No valid image directory '{image_subdir}' found under any subdirectory of {scene_path}"
    )


def list_images(img_dir):
    """列出目录中的所有图像文件"""
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in extensions:
        images.extend(list(img_dir.glob(f"*{ext}")))
        images.extend(list(img_dir.glob(f"*{ext.upper()}")))
    return sorted(images, key=natural_key)


def xyxy_to_cxcywh_normalized(bbox, img_width, img_height):
    """将 [x0,y0,x1,y1] 转成 [cx,cy,w,h] 归一化格式"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox, expected [x0,y0,x1,y1], got: {bbox}")

    x0, y0, x1, y1 = bbox

    center_x = (x0 + x1) / 2.0 / img_width
    center_y = (y0 + y1) / 2.0 / img_height
    width = (x1 - x0) / img_width
    height = (y1 - y0) / img_height

    return [center_x, center_y, width, height]


def normalize_frame_bboxes(frame_bboxes):
    """
    将单帧 bbox 统一成:
    [
      {"prompt_name": str, "bbox": [x0,y0,x1,y1]},
      ...
    ]
    """
    normalized = []

    if not frame_bboxes:
        return normalized

    if isinstance(frame_bboxes, dict):
        for prompt_name, bbox in frame_bboxes.items():
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                normalized.append(
                    {
                        "prompt_name": str(prompt_name),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    }
                )
            else:
                print(f"Warning: skip invalid bbox for prompt '{prompt_name}': {bbox}")
        return normalized

    if isinstance(frame_bboxes, list):
        for i, item in enumerate(frame_bboxes):
            if isinstance(item, (list, tuple)) and len(item) == 4:
                normalized.append(
                    {
                        "prompt_name": f"prompt_{i}",
                        "bbox": [float(item[0]), float(item[1]), float(item[2]), float(item[3])],
                    }
                )
            elif isinstance(item, dict) and "bbox" in item:
                bbox = item["bbox"]
                prompt_name = item.get("prompt_name", f"prompt_{i}")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    normalized.append(
                        {
                            "prompt_name": str(prompt_name),
                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        }
                    )
                else:
                    print(f"Warning: skip invalid bbox item at index {i}: {item}")
            else:
                print(f"Warning: skip invalid bbox item at index {i}: {item}")
        return normalized

    raise TypeError(f"Unsupported frame bbox type: {type(frame_bboxes)}")


def tensor_mask_to_uint8(mask_tensor):
    """
    将 SAM3 输出的 mask tensor 转成 uint8 二值图
    输出取值: 0 / 255
    """
    mask_np = mask_tensor.detach().cpu().numpy().squeeze()
    mask_bin = (mask_np > 0).astype(np.uint8) * 255
    return mask_bin


def process_frame_with_bboxes(processor, image, frame_bboxes, confidence_threshold=0.5):
    """
    处理单帧图像和多个 bbox
    返回:
      saved_masks: list[np.ndarray(H,W), uint8]
      result_infos: list[dict]
    """
    prompt_items = normalize_frame_bboxes(frame_bboxes)
    if not prompt_items:
        return [], []

    state = processor.set_image(image)
    img_width, img_height = image.size

    saved_masks = []
    result_infos = []

    for item in prompt_items:
        prompt_name = item["prompt_name"]
        bbox = item["bbox"]

        try:
            normalized_bbox = xyxy_to_cxcywh_normalized(bbox, img_width, img_height)
        except Exception as e:
            print(f"Warning: failed to normalize bbox for '{prompt_name}': {bbox}, error: {e}")
            continue

        processor.reset_all_prompts(state)
        state = processor.add_geometric_prompt(normalized_bbox, True, state)

        masks = state.get("masks", [])
        boxes = state.get("boxes", [])
        scores = state.get("scores", [])

        if len(masks) == 0 or len(scores) == 0:
            continue

        for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            score_val = float(score)
            if score_val <= confidence_threshold:
                continue

            mask_uint8 = tensor_mask_to_uint8(mask)
            saved_masks.append(mask_uint8)

            pred_box_xyxy = None
            if box is not None:
                pred_box_xyxy = box.detach().cpu().numpy().tolist()

            result_infos.append(
                {
                    "prompt_name": prompt_name,
                    "input_bbox": bbox,
                    "output_index": idx,
                    "score": score_val,
                    "pred_box_xyxy": pred_box_xyxy,
                }
            )

    return saved_masks, result_infos


def save_mask(mask_array, save_path):
    """
    保存单张 mask
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array).save(save_path)


def merge_masks(mask_list):
    """
    将多张 uint8 mask 做并集 merge
    """
    if not mask_list:
        return None

    merged = np.zeros_like(mask_list[0], dtype=np.uint8)
    for m in mask_list:
        merged = np.maximum(merged, m)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Batch process frames with bbox prompts using SAM3")
    parser.add_argument("--bbox-json", required=True, help="JSON file containing bbox data for each frame")
    parser.add_argument("--scene-dir", required=True, help="Scene directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--image-subdir", default="hires_wide", help="Image subdirectory name")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument(
        "--save-merged-mask",
        action="store_true",
        help="Whether to save merged union mask for each frame",
    )
    args = parser.parse_args()

    bbox_json_path = Path(args.bbox_json)
    if not bbox_json_path.exists():
        raise FileNotFoundError(f"Bbox JSON file not found: {bbox_json_path}")

    scene_dir = Path(args.scene_dir)
    output_root = Path(args.output_dir)
    scene_id = scene_dir.name

    print(f"Loading bbox data from: {bbox_json_path}")
    with open(bbox_json_path, "r") as f:
        bbox_data = json.load(f)

    print(f"Finding image directory in scene: {scene_dir}")
    img_dir = find_first_sequence_image_dir(scene_dir, args.image_subdir)
    print(f"Using image directory: {img_dir}")

    images = list_images(img_dir)
    print(f"Found {len(images)} images")
    if len(images) == 0:
        raise RuntimeError(f"No images found in: {img_dir}")

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print(f"Model loaded on device: {device}")

    scene_out_dir = output_root / scene_id
    masks_root = scene_out_dir / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_prompts = 0
    total_masks = 0
    processed_frames = []

    num_frames = len(images)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    print(f"Processing {num_frames} frames...")

    for frame_id in tqdm(range(num_frames), desc="Processing frames"):
        frame_key = str(frame_id)
        frame_bboxes = bbox_data.get(frame_key, [])

        frame_mask_dir = masks_root / str(frame_id)
        frame_mask_dir.mkdir(parents=True, exist_ok=True)

        image_path = images[frame_id]
        image = Image.open(image_path).convert("RGB")

        prompt_items = normalize_frame_bboxes(frame_bboxes)
        masks, result_infos = process_frame_with_bboxes(
            processor=processor,
            image=image,
            frame_bboxes=frame_bboxes,
            confidence_threshold=args.confidence,
        )

        for i, mask_array in enumerate(masks):
            mask_path = frame_mask_dir / f"mask{i}.png"
            save_mask(mask_array, mask_path)

        merged_mask_path = None
        if args.save_merged_mask and len(masks) > 0:
            merged = merge_masks(masks)
            merged_mask_path = frame_mask_dir / "merged_mask.png"
            save_mask(merged, merged_mask_path)

        total_frames += 1
        total_prompts += len(prompt_items)
        total_masks += len(masks)

        processed_frames.append(
            {
                "frame_id": frame_id,
                "image_path": str(image_path),
                "num_bbox_prompts": len(prompt_items),
                "num_generated_masks": len(masks),
                "bbox_prompts_raw": frame_bboxes,
                "bbox_prompts_normalized": prompt_items,
                "generated_results": result_infos,
                "mask_dir": str(frame_mask_dir),
                "merged_mask_path": str(merged_mask_path) if merged_mask_path is not None else None,
            }
        )

    summary = {
        "input_bbox_json": str(bbox_json_path),
        "scene_directory": str(scene_dir),
        "image_directory": str(img_dir),
        "output_directory": str(scene_out_dir),
        "scene_id": scene_id,
        "confidence_threshold": args.confidence,
        "save_merged_mask": args.save_merged_mask,
        "statistics": {
            "total_frames_processed": total_frames,
            "total_bbox_prompts": total_prompts,
            "total_generated_masks": total_masks,
            "average_masks_per_frame": total_masks / total_frames if total_frames > 0 else 0.0,
            "average_prompts_per_frame": total_prompts / total_frames if total_frames > 0 else 0.0,
        },
        "processed_frames": processed_frames,
    }

    summary_path = scene_out_dir / "batch_processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Processing Complete ===")
    print(f"Frames processed: {total_frames}")
    print(f"Total bbox prompts: {total_prompts}")
    print(f"Total generated masks: {total_masks}")
    print(f"Average masks per frame: {total_masks / total_frames if total_frames > 0 else 0:.2f}")
    print(f"Masks root: {masks_root}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()