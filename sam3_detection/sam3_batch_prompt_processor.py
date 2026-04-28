#!/usr/bin/env python3
"""
批量处理 bbox / point prompt 的 SAM3 脚本

支持三种输入数据格式：

1) bbox 模式 - 简单bbox prompt格式
{
  "30": [
    [30, 0, 86, 179],
    [100, 120, 180, 240]
  ]
}

或
{
  "30": {
    "Close the bedroom door": [30, 0, 86, 179],
    "Open the first drawer": [100, 120, 180, 240]
  }
}

2) point 模式 - 点击prompt格式
{
  "30": {
    "INT__door_handle__000__area10515.png": [73, 117],
    "INT__drawer_handle__000__area1287.png": [725, 138]
  }
}

3) grounding 模式 - 从完整grounding结果中提取bbox
直接输入完整的grounding detection结果JSON，脚本自动提取label=1的bbox作为prompt。
注意：grounding summary 里的 bbox 是 0~1000 归一化坐标，脚本会按当前图像尺寸自动转换为像素坐标。
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def natural_key(p):
    parts = re.split(r"(\d+)", str(p))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def find_first_sequence_image_dir(scene_dir, image_subdir="hires_wide"):
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
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in extensions:
        images.extend(list(img_dir.glob(f"*{ext}")))
        images.extend(list(img_dir.glob(f"*{ext.upper()}")))
    return sorted(images, key=natural_key)


def xyxy_to_cxcywh_normalized(bbox, img_width, img_height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox, expected [x0,y0,x1,y1], got: {bbox}")

    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2.0 / img_width
    center_y = (y0 + y1) / 2.0 / img_height
    width = (x1 - x0) / img_width
    height = (y1 - y0) / img_height
    return [center_x, center_y, width, height]


def point_xy_to_normalized(point, img_width, img_height):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Invalid point, expected [x,y], got: {point}")
    x, y = point
    return [float(x) / img_width, float(y) / img_height]


def point_to_small_bbox_xyxy(point, img_width, img_height, point_box_radius_px=8):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Invalid point, expected [x,y], got: {point}")

    x, y = float(point[0]), float(point[1])
    x0 = max(0.0, x - point_box_radius_px)
    y0 = max(0.0, y - point_box_radius_px)
    x1 = min(float(img_width - 1), x + point_box_radius_px)
    y1 = min(float(img_height - 1), y + point_box_radius_px)
    return [x0, y0, x1, y1]


def grounding_bbox_to_pixel_xyxy(bbox, img_width, img_height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid grounding bbox: {bbox}")
    x0, y0, x1, y1 = bbox
    
    # Detect if bbox is already in pixel coordinates (not normalized 0-1000)
    # If all coordinates are within image bounds, assume they're already pixels
    if (0 <= x0 <= img_width and 0 <= y0 <= img_height and 
        0 <= x1 <= img_width and 0 <= y1 <= img_height):
        print(f"Bbox {bbox} appears to be in pixel coordinates, using directly")
        return [float(x0), float(y0), float(x1), float(y1)]
    else:
        # Assume 0-1000 normalized coordinates
        print(f"Bbox {bbox} appears to be normalized (0-1000), converting to pixels")
        return [
            float(x0) / 1000.0 * img_width,
            float(y0) / 1000.0 * img_height,
            float(x1) / 1000.0 * img_width,
            float(y1) / 1000.0 * img_height,
        ]


def load_grounding_prompts(grounding_json_path):
    """
    从grounding JSON文件中提取label=1的bbox prompts

    返回格式: {frame_index_str: [[x0,y0,x1,y1], ...]}
    注意：这里返回的仍然是原始 0~1000 坐标，后续会在拿到图像尺寸后转成像素坐标。
    """
    with open(grounding_json_path, "r") as f:
        data = json.load(f)

    bbox_prompts = {}
    total_bboxes = 0
    label1_bboxes = 0

    # Handle both old format (with scenes) and new format (direct frames)
    frames_to_process = []
    
    if "scenes" in data:
        # Old format: data.scenes[].frames[]
        for scene in data.get("scenes", []):
            frames_to_process.extend(scene.get("frames", []))
    elif "frames" in data:
        # New format: data.frames[]
        frames_to_process = data.get("frames", [])
    else:
        print("Warning: No 'scenes' or 'frames' found in JSON file")
        return bbox_prompts

    for frame in frames_to_process:
        frame_index = frame.get("frame_index")

        if frame_index is None:
            continue

        frame_bboxes = []
        objects = frame.get("objects", {})
        for _, obj_list in objects.items():
            for obj in obj_list:
                bbox = obj.get("bbox")
                label = obj.get("label")

                if bbox is None or label is None:
                    continue

                total_bboxes += 1
                if label == 1:
                    label1_bboxes += 1
                    frame_bboxes.append(bbox)

        if frame_bboxes:
            bbox_prompts[str(frame_index)] = frame_bboxes

    print(
        f"Grounding JSON stats: {total_bboxes} total bboxes, "
        f"{label1_bboxes} label=1 bboxes, {len(bbox_prompts)} frames with prompts"
    )
    return bbox_prompts


def normalize_frame_prompts(frame_prompts, prompt_type):
    normalized = []

    if not frame_prompts:
        return normalized

    if isinstance(frame_prompts, dict):
        for prompt_name, prompt in frame_prompts.items():
            if prompt_type == "bbox":
                if isinstance(prompt, (list, tuple)) and len(prompt) == 4:
                    normalized.append(
                        {
                            "prompt_name": str(prompt_name),
                            "prompt_type": "bbox",
                            "prompt": [float(prompt[0]), float(prompt[1]), float(prompt[2]), float(prompt[3])],
                        }
                    )
                else:
                    print(f"Warning: skip invalid bbox for prompt '{prompt_name}': {prompt}")
            elif prompt_type == "point":
                if isinstance(prompt, (list, tuple)) and len(prompt) == 2:
                    normalized.append(
                        {
                            "prompt_name": str(prompt_name),
                            "prompt_type": "point",
                            "prompt": [float(prompt[0]), float(prompt[1])],
                        }
                    )
                else:
                    print(f"Warning: skip invalid point for prompt '{prompt_name}': {prompt}")
        return normalized

    if isinstance(frame_prompts, list):
        for i, item in enumerate(frame_prompts):
            if prompt_type in {"bbox", "grounding"}:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    normalized.append(
                        {
                            "prompt_name": f"prompt_{i}",
                            "prompt_type": "bbox",
                            "prompt": [float(item[0]), float(item[1]), float(item[2]), float(item[3])],
                        }
                    )
                elif isinstance(item, dict) and "bbox" in item:
                    bbox = item["bbox"]
                    prompt_name = item.get("prompt_name", f"prompt_{i}")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        normalized.append(
                            {
                                "prompt_name": str(prompt_name),
                                "prompt_type": "bbox",
                                "prompt": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            }
                        )
                    else:
                        print(f"Warning: skip invalid bbox item at index {i}: {item}")
                else:
                    print(f"Warning: skip invalid bbox item at index {i}: {item}")

            elif prompt_type == "point":
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    normalized.append(
                        {
                            "prompt_name": f"prompt_{i}",
                            "prompt_type": "point",
                            "prompt": [float(item[0]), float(item[1])],
                        }
                    )
                elif isinstance(item, dict) and "point" in item:
                    point = item["point"]
                    prompt_name = item.get("prompt_name", f"prompt_{i}")
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        normalized.append(
                            {
                                "prompt_name": str(prompt_name),
                                "prompt_type": "point",
                                "prompt": [float(point[0]), float(point[1])],
                            }
                        )
                    else:
                        print(f"Warning: skip invalid point item at index {i}: {item}")
                else:
                    print(f"Warning: skip invalid point item at index {i}: {item}")

        return normalized

    raise TypeError(f"Unsupported frame prompt type: {type(frame_prompts)}")


def mask_area(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy().squeeze()
    return int((mask_np > 0).sum())


def mask_tensor_to_uint8(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy().squeeze()
    mask_bin = (mask_np > 0).astype(np.uint8) * 255
    return mask_bin


def try_add_point_prompt(processor, point_xy, state):
    candidate_methods = [
        "add_point_prompt",
        "add_points_prompt",
        "add_points",
    ]

    for method_name in candidate_methods:
        if hasattr(processor, method_name):
            method = getattr(processor, method_name)
            trials = [
                lambda: method(point_xy, True, state),
                lambda: method([point_xy], [True], state),
                lambda: method([point_xy], state),
                lambda: method(point_xy, state),
            ]

            for fn in trials:
                try:
                    new_state = fn()
                    return new_state, True
                except TypeError:
                    continue
                except Exception:
                    continue

    return state, False


def process_frame_with_prompts(
    processor,
    image,
    frame_prompts,
    prompt_type="bbox",
    confidence_threshold=0.5,
    point_box_radius_px=8,
    dual_mask=False,
):
    prompt_items = normalize_frame_prompts(frame_prompts, prompt_type=prompt_type)
    if not prompt_items:
        if dual_mask:
            return ([], []), []
        return [], []

    state = processor.set_image(image)
    img_width, img_height = image.size

    if dual_mask:
        best_score_masks = []
        smallest_area_masks = []
    else:
        selected_masks_uint8 = []
    result_infos = []

    for item in prompt_items:
        prompt_name = item["prompt_name"]
        prompt_value = item["prompt"]

        processor.reset_all_prompts(state)

        used_prompt_mode = prompt_type
        normalized_prompt = None

        try:
            if prompt_type == "bbox":
                normalized_prompt = xyxy_to_cxcywh_normalized(prompt_value, img_width, img_height)
                state = processor.add_geometric_prompt(normalized_prompt, True, state)

            elif prompt_type == "point":
                normalized_point = point_xy_to_normalized(prompt_value, img_width, img_height)
                state_candidate, ok = try_add_point_prompt(processor, normalized_point, state)
                if ok:
                    state = state_candidate
                    used_prompt_mode = "point_native"
                    normalized_prompt = normalized_point
                else:
                    fallback_bbox_xyxy = point_to_small_bbox_xyxy(
                        prompt_value,
                        img_width,
                        img_height,
                        point_box_radius_px=point_box_radius_px,
                    )
                    normalized_prompt = xyxy_to_cxcywh_normalized(fallback_bbox_xyxy, img_width, img_height)
                    state = processor.add_geometric_prompt(normalized_prompt, True, state)
                    used_prompt_mode = "point_as_small_bbox"

            else:
                raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        except Exception as e:
            print(f"Warning: failed to apply prompt for '{prompt_name}': {prompt_value}, error: {e}")
            continue

        masks = state.get("masks", [])
        boxes = state.get("boxes", [])
        scores = state.get("scores", [])

        if len(masks) == 0 or len(scores) == 0:
            continue

        candidates = []
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            score_val = float(score)
            if score_val <= confidence_threshold:
                continue
            area = mask_area(mask)
            if area <= 0:
                continue
            candidates.append((area, idx, score_val))

        if not candidates:
            continue

        if dual_mask:
            best_score_candidate = max(candidates, key=lambda x: x[2])
            best_score_area, best_score_idx, best_score_val = best_score_candidate

            smallest_area_candidate = min(candidates, key=lambda x: x[0])
            smallest_area_val, smallest_area_idx, smallest_area_score = smallest_area_candidate

            best_score_mask = masks[best_score_idx]
            smallest_area_mask = masks[smallest_area_idx]

            best_score_masks.append(mask_tensor_to_uint8(best_score_mask))
            smallest_area_masks.append(mask_tensor_to_uint8(smallest_area_mask))

            best_box = boxes[best_score_idx] if best_score_idx < len(boxes) else None
        else:
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_area, best_idx, best_score = candidates[0]

            best_mask = masks[best_idx]
            best_box = boxes[best_idx] if best_idx < len(boxes) else None

            selected_masks_uint8.append(mask_tensor_to_uint8(best_mask))

        box_xyxy = None
        if best_box is not None:
            try:
                box_xyxy = best_box.detach().cpu().numpy().tolist()
            except Exception:
                box_xyxy = None

        if dual_mask:
            result_infos.append(
                {
                    "prompt_name": prompt_name,
                    "input_prompt_type": prompt_type,
                    "used_prompt_mode": used_prompt_mode,
                    "input_prompt": prompt_value,
                    "normalized_prompt": normalized_prompt,
                    "selection_rule": "dual_output_best_score_and_smallest_area",
                    "best_score_info": {
                        "area": best_score_area,
                        "index": best_score_idx,
                        "score": best_score_val,
                    },
                    "smallest_area_info": {
                        "area": smallest_area_val,
                        "index": smallest_area_idx,
                        "score": smallest_area_score,
                    },
                    "pred_box_xyxy": box_xyxy,
                    "num_candidates_before_filter": len(masks),
                    "num_candidates_after_score_filter": len(candidates),
                }
            )
        else:
            result_infos.append(
                {
                    "prompt_name": prompt_name,
                    "input_prompt_type": prompt_type,
                    "used_prompt_mode": used_prompt_mode,
                    "input_prompt": prompt_value,
                    "normalized_prompt": normalized_prompt,
                    "selected_output_index": best_idx,
                    "selection_rule": "smallest_area_above_confidence_threshold",
                    "selected_mask_area": best_area,
                    "score": best_score,
                    "pred_box_xyxy": box_xyxy,
                    "num_candidates_before_filter": len(masks),
                    "num_candidates_after_score_filter": len(candidates),
                }
            )

    if dual_mask:
        return (best_score_masks, smallest_area_masks), result_infos
    return selected_masks_uint8, result_infos


def create_visualization(image, mask_arrays, bboxes, vis_path, title=""):
    vis_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, mask_array in enumerate(mask_arrays):
        if mask_array is None:
            continue

        color = colors[i % len(colors)]
        mask_bool = mask_array > 0
        colored_mask = np.zeros_like(img_array)
        colored_mask[mask_bool] = color

        alpha = 0.4
        img_array = img_array.astype(np.float32)
        colored_mask = colored_mask.astype(np.float32)
        img_array[mask_bool] = (1 - alpha) * img_array[mask_bool] + alpha * colored_mask[mask_bool]

    img_array = img_array.astype(np.uint8)
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)

    for i, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, y0 - 20), f"Prompt_{i}", fill=color)

    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))

    img_pil.save(vis_path)


def save_mask(mask_array, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array).save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Batch process frames with bbox or point prompts using SAM3")
    parser.add_argument("--prompt-json", required=True, help="JSON file containing prompt data for each frame")
    parser.add_argument(
        "--prompt-type",
        required=True,
        choices=["bbox", "point", "grounding"],
        help="Prompt type in JSON: bbox, point, or grounding",
    )
    parser.add_argument("--scene-dir", required=True, help="Scene directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--image-subdir", default="hires_wide", help="Image subdirectory name")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--point-box-radius",
        type=int,
        default=8,
        help="When point native prompt is unavailable, convert point to a small bbox with this radius in pixels",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process; -1 means all frames",
    )
    parser.add_argument(
        "--dual-mask",
        action="store_true",
        help="Enable dual mask output: best score + smallest area",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images with masks and bboxes overlaid",
    )
    args = parser.parse_args()

    prompt_json_path = Path(args.prompt_json)
    if not prompt_json_path.exists():
        raise FileNotFoundError(f"Prompt JSON file not found: {prompt_json_path}")

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    print(f"Finding image directory in scene: {scene_dir}")
    img_dir = find_first_sequence_image_dir(scene_dir, args.image_subdir)
    images = list_images(img_dir)

    if args.prompt_type == "grounding":
        print(f"Loading grounding data from: {prompt_json_path}")
        prompt_data = load_grounding_prompts(prompt_json_path)
        grounding_mode = True
    else:
        print(f"Loading prompt data from: {prompt_json_path}")
        with open(prompt_json_path, "r") as f:
            prompt_data = json.load(f)
        grounding_mode = False

    print(f"Using image directory: {img_dir}")
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

    scene_id = scene_dir.name
    scene_out_dir = output_dir / scene_id

    if args.dual_mask:
        best_score_root = scene_out_dir / "masks_best_score"
        smallest_area_root = scene_out_dir / "masks_smallest_area"
        best_score_root.mkdir(parents=True, exist_ok=True)
        smallest_area_root.mkdir(parents=True, exist_ok=True)
        masks_root = None
    else:
        masks_root = scene_out_dir / "masks"
        masks_root.mkdir(parents=True, exist_ok=True)
        best_score_root = None
        smallest_area_root = None

    if args.save_vis:
        if args.dual_mask:
            vis_best_score_root = scene_out_dir / "vis_best_score"
            vis_smallest_area_root = scene_out_dir / "vis_smallest_area"
            vis_best_score_root.mkdir(parents=True, exist_ok=True)
            vis_smallest_area_root.mkdir(parents=True, exist_ok=True)
            vis_root = None
        else:
            vis_root = scene_out_dir / "vis"
            vis_root.mkdir(parents=True, exist_ok=True)
            vis_best_score_root = None
            vis_smallest_area_root = None

    total_frames = 0
    total_prompts = 0
    total_masks = 0
    processed_frames = []

    num_frames = len(images)
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    frame_ids_to_process = list(range(num_frames))

    if grounding_mode:
        frame_keys_to_process = list(prompt_data.keys())
        print(f"Grounding mode: Processing {len(frame_keys_to_process)} frames with prompts")

        frame_to_image = {}
        for frame_key in frame_keys_to_process:
            frame_idx = int(frame_key)
            if frame_idx < len(images):
                frame_to_image[frame_key] = images[frame_idx]
            else:
                print(f"Warning: frame_index {frame_idx} exceeds image sequence length {len(images)}")

        print(f"Successfully mapped {len(frame_to_image)}/{len(frame_keys_to_process)} frames to images")

        for frame_key in tqdm(frame_keys_to_process, desc="Processing frames"):
            if frame_key not in frame_to_image:
                continue

            image_path = frame_to_image[frame_key]
            frame_prompts = prompt_data.get(frame_key, [])
            output_frame_id = frame_key

            if args.dual_mask:
                best_score_frame_dir = best_score_root / str(output_frame_id)
                smallest_area_frame_dir = smallest_area_root / str(output_frame_id)
                best_score_frame_dir.mkdir(parents=True, exist_ok=True)
                smallest_area_frame_dir.mkdir(parents=True, exist_ok=True)
            else:
                frame_mask_dir = masks_root / str(output_frame_id)
                frame_mask_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size

            pixel_frame_prompts = [
                grounding_bbox_to_pixel_xyxy(bbox, img_width, img_height)
                for bbox in frame_prompts
            ]

            prompt_items = normalize_frame_prompts(pixel_frame_prompts, prompt_type="bbox")

            masks_result, result_infos = process_frame_with_prompts(
                processor=processor,
                image=image,
                frame_prompts=pixel_frame_prompts,
                prompt_type="bbox",
                confidence_threshold=args.confidence,
                point_box_radius_px=args.point_box_radius,
                dual_mask=args.dual_mask,
            )

            if args.dual_mask:
                best_score_masks, smallest_area_masks = masks_result

                for i, mask_array in enumerate(best_score_masks):
                    save_mask(mask_array, best_score_frame_dir / f"best_score_mask{i}.png")

                for i, mask_array in enumerate(smallest_area_masks):
                    save_mask(mask_array, smallest_area_frame_dir / f"smallest_area_mask{i}.png")

                if args.save_vis:
                    input_bboxes = [item["prompt"] for item in prompt_items]
                    create_visualization(
                        image,
                        best_score_masks,
                        input_bboxes,
                        vis_best_score_root / f"vis_{output_frame_id}.jpg",
                        f"Frame {output_frame_id} - Best Score Masks",
                    )
                    create_visualization(
                        image,
                        smallest_area_masks,
                        input_bboxes,
                        vis_smallest_area_root / f"vis_{output_frame_id}.jpg",
                        f"Frame {output_frame_id} - Smallest Area Masks",
                    )
            else:
                masks_uint8 = masks_result
                for i, mask_array in enumerate(masks_uint8):
                    save_mask(mask_array, frame_mask_dir / f"mask{i}.png")

                if args.save_vis:
                    input_bboxes = [item["prompt"] for item in prompt_items]
                    create_visualization(
                        image,
                        masks_uint8,
                        input_bboxes,
                        vis_root / f"vis_{output_frame_id}.jpg",
                        f"Frame {output_frame_id} - Masks",
                    )

            total_frames += 1
            total_prompts += len(prompt_items)
            total_masks += len(best_score_masks) + len(smallest_area_masks) if args.dual_mask else len(masks_uint8)

            frame_info = {
                "frame_id": output_frame_id,
                "image_path": str(image_path),
                "prompt_type": args.prompt_type,
                "num_prompts": len(prompt_items),
                "prompts_raw": frame_prompts,
                "prompts_pixel_xyxy": pixel_frame_prompts,
                "prompts_normalized": prompt_items,
                "generated_results": result_infos,
            }

            if args.dual_mask:
                frame_info.update(
                    {
                        "num_best_score_masks": len(best_score_masks),
                        "num_smallest_area_masks": len(smallest_area_masks),
                        "best_score_mask_dir": str(best_score_frame_dir),
                        "smallest_area_mask_dir": str(smallest_area_frame_dir),
                    }
                )
            else:
                frame_info.update(
                    {
                        "num_generated_masks": len(masks_uint8),
                        "mask_dir": str(frame_mask_dir),
                    }
                )
            processed_frames.append(frame_info)

    else:
        if len(frame_ids_to_process) == 0:
            print("No frames to process.")
        else:
            print(f"Processing {len(frame_ids_to_process)} frames from 0 to {frame_ids_to_process[-1]}...")

            for frame_id in tqdm(frame_ids_to_process, desc="Processing frames"):
                frame_key = str(frame_id)
                frame_prompts = prompt_data.get(frame_key, {} if args.prompt_type == "point" else [])
                image_path = images[frame_id]

                if args.dual_mask:
                    best_score_frame_dir = best_score_root / str(frame_id)
                    smallest_area_frame_dir = smallest_area_root / str(frame_id)
                    best_score_frame_dir.mkdir(parents=True, exist_ok=True)
                    smallest_area_frame_dir.mkdir(parents=True, exist_ok=True)
                else:
                    frame_mask_dir = masks_root / str(frame_id)
                    frame_mask_dir.mkdir(parents=True, exist_ok=True)

                image = Image.open(image_path).convert("RGB")
                prompt_items = normalize_frame_prompts(frame_prompts, prompt_type=args.prompt_type)

                masks_result, result_infos = process_frame_with_prompts(
                    processor=processor,
                    image=image,
                    frame_prompts=frame_prompts,
                    prompt_type=args.prompt_type,
                    confidence_threshold=args.confidence,
                    point_box_radius_px=args.point_box_radius,
                    dual_mask=args.dual_mask,
                )

                if args.dual_mask:
                    best_score_masks, smallest_area_masks = masks_result
                    for i, mask_array in enumerate(best_score_masks):
                        save_mask(mask_array, best_score_frame_dir / f"best_score_mask{i}.png")
                    for i, mask_array in enumerate(smallest_area_masks):
                        save_mask(mask_array, smallest_area_frame_dir / f"smallest_area_mask{i}.png")
                    if args.save_vis:
                        input_bboxes = [item["prompt"] for item in prompt_items]
                        create_visualization(
                            image,
                            best_score_masks,
                            input_bboxes,
                            vis_best_score_root / f"vis_{frame_id}.jpg",
                            f"Frame {frame_id} - Best Score Masks",
                        )
                        create_visualization(
                            image,
                            smallest_area_masks,
                            input_bboxes,
                            vis_smallest_area_root / f"vis_{frame_id}.jpg",
                            f"Frame {frame_id} - Smallest Area Masks",
                        )
                else:
                    masks_uint8 = masks_result
                    for i, mask_array in enumerate(masks_uint8):
                        save_mask(mask_array, frame_mask_dir / f"mask{i}.png")
                    if args.save_vis:
                        input_bboxes = [item["prompt"] for item in prompt_items]
                        create_visualization(
                            image,
                            masks_uint8,
                            input_bboxes,
                            vis_root / f"vis_{frame_id}.jpg",
                            f"Frame {frame_id} - Masks",
                        )

                total_frames += 1
                total_prompts += len(prompt_items)
                total_masks += len(best_score_masks) + len(smallest_area_masks) if args.dual_mask else len(masks_uint8)

                frame_info = {
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "prompt_type": args.prompt_type,
                    "num_prompts": len(prompt_items),
                    "prompts_raw": frame_prompts,
                    "prompts_normalized": prompt_items,
                    "generated_results": result_infos,
                }

                if args.dual_mask:
                    frame_info.update(
                        {
                            "num_best_score_masks": len(best_score_masks),
                            "num_smallest_area_masks": len(smallest_area_masks),
                            "best_score_mask_dir": str(best_score_frame_dir),
                            "smallest_area_mask_dir": str(smallest_area_frame_dir),
                        }
                    )
                else:
                    frame_info.update(
                        {
                            "num_generated_masks": len(masks_uint8),
                            "mask_dir": str(frame_mask_dir),
                        }
                    )
                processed_frames.append(frame_info)

    summary = {
        "input_prompt_json": str(prompt_json_path),
        "prompt_type": args.prompt_type,
        "scene_directory": str(scene_dir),
        "image_directory": str(img_dir),
        "output_directory": str(output_dir),
        "dual_mask_mode": args.dual_mask,
        "confidence_threshold": args.confidence,
        "point_fallback_policy": (
            "try native point prompt first; if unavailable, convert point to a small bbox"
            if args.prompt_type == "point"
            else None
        ),
        "point_box_radius": args.point_box_radius if args.prompt_type == "point" else None,
        "frame_policy": "process every frame from 0 to the last available image",
        "statistics": {
            "total_frames_processed": total_frames,
            "total_prompts": total_prompts,
            "total_generated_masks": total_masks,
            "average_masks_per_frame": total_masks / total_frames if total_frames > 0 else 0.0,
            "average_prompts_per_frame": total_prompts / total_frames if total_frames > 0 else 0.0,
        },
        "processed_frames": processed_frames,
    }

    if args.dual_mask:
        summary.update(
            {
                "best_score_masks_root": str(best_score_root),
                "smallest_area_masks_root": str(smallest_area_root),
                "selection_rule": "dual output: best score mask + smallest area mask above confidence threshold",
            }
        )
    else:
        summary.update(
            {
                "masks_root": str(masks_root),
                "selection_rule": "for each prompt keep only the smallest-area mask above confidence threshold",
            }
        )

    summary_path = output_dir / "batch_processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Processing Complete ===")
    print(f"Frames processed: {total_frames}")
    print(f"Total prompts: {total_prompts}")
    print(f"Total generated masks: {total_masks}")
    print(f"Average masks per frame: {total_masks / total_frames if total_frames > 0 else 0:.2f}")
    if args.dual_mask:
        print(f"Best score masks root: {best_score_root}")
        print(f"Smallest area masks root: {smallest_area_root}")
    else:
        print(f"Masks root: {masks_root}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
