#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metric.py

用途：
- 输入 pred.json (annot_id -> {mask_png, frame, score})，把“全局 laser scan 点云”投影到指定帧，
  取落在 mask 内部的“原始点云索引 indices”，生成 pred3d。
- （可选）用 GT annotations.json 的 indices 计算 mAP/AP50/AP25 + mIoU_aligned_by_annot_id。
- （可选）保存“点云投影到该帧”的可视化图片（黑底，点用 PLY 原颜色）。

关键修正（你已验证 B_Tcw inside 很大）：
- traj 里的 pose 采用 **T_cw (camera <- world)**：
    p_c = R * p_w + t
  之前用 T_wc 的公式会导致 indices 全空、指标全 0。

可选参数：
- --no-depth-filter：去掉 depth 一致性过滤（只用 mask hit）
- --proj-vis-dir：保存投影可视化（只保存 B_Tcw，不再存 A_Twc）
- --depth-scale：depth_raw 转米用的 scale（默认 1000.0，适用于 uint16 mm）
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

sys.path.append(os.getcwd())

# Support both running from project root (scripts.proj_3d) and local folder (proj_3d.py).
try:
    from proj_3d import (  # type: ignore
        Intrinsics,
        find_first_sequence_dir,
        find_intr_dir,
        find_traj_path,
        list_depth_pngs,
        load_depth_png,
        load_intrinsics_txt,
        read_traj_Twc,
    )
except Exception:
    from scripts.proj_3d import (  # type: ignore
        Intrinsics,
        find_first_sequence_dir,
        find_intr_dir,
        find_traj_path,
        list_depth_pngs,
        load_depth_png,
        load_intrinsics_txt,
        read_traj_Twc,
    )


# -------------------------
# Small utils
# -------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _safe_read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _find_unique_json(scene_dir: Path, must_contain_key: str) -> Path:
    cands = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if not cands:
        raise FileNotFoundError(f"No .json files under: {scene_dir}")

    name_hits = [p for p in cands if must_contain_key in p.name.lower()]
    if len(name_hits) == 1:
        return name_hits[0]

    content_hits = []
    for p in cands:
        try:
            obj = _safe_read_json(p)
            if must_contain_key in obj:
                content_hits.append(p)
        except Exception:
            continue

    if len(content_hits) == 1:
        return content_hits[0]

    raise ValueError(
        f"Ambiguous json for key='{must_contain_key}' in {scene_dir}. "
        f"Candidates={sorted([x.name for x in cands])}, "
        f"name_hits={sorted([x.name for x in name_hits])}, "
        f"content_hits={sorted([x.name for x in content_hits])}"
    )


def _resolve_scene_jsons(data_root: Path, scene_id: Union[int, str]) -> Tuple[Path, Path]:
    sid = str(scene_id)
    scene_dir = data_root / sid
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    desc_json = _find_unique_json(scene_dir, "descriptions")
    ann_json = _find_unique_json(scene_dir, "annotations")
    return desc_json, ann_json


# -------------------------
# PLY loading: points + original rgb (if exists)
# -------------------------

def _load_laser_scan_points_and_rgb(scene_dir: Path, scene_id: Union[int, str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      pts: (N,3) float32
      rgb: (N,3) uint8 if present in PLY, else None
    """
    sid = str(scene_id)
    ply = scene_dir / f"{sid}_laser_scan.ply"
    if not ply.exists():
        hits = sorted(list(scene_dir.glob("*_laser_scan.ply")), key=lambda p: _natural_key(p.name))
        if not hits:
            raise FileNotFoundError(f"Cannot find laser scan ply under {scene_dir} (expected {ply.name})")
        ply = hits[0]

    # Try open3d first if available.
    try:
        import open3d as o3d  # type: ignore
        pcd = o3d.io.read_point_cloud(str(ply))
        pts = np.asarray(pcd.points, dtype=np.float32)
        rgb = None
        if pcd.has_colors():
            c = np.asarray(pcd.colors, dtype=np.float32)  # 0~1
            rgb = (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if pts.size != 0:
            return pts, rgb
    except Exception:
        pass

    # Fallback minimal PLY reader for xyz + optional rgb
    fmt = None
    vertex_count = None
    props: List[Tuple[str, str]] = []
    in_vertex = False

    def _dtype_from_type(t: str) -> np.dtype:
        m = {
            "char": "i1", "uchar": "u1",
            "int8": "i1", "uint8": "u1",
            "short": "i2", "ushort": "u2",
            "int16": "i2", "uint16": "u2",
            "int": "i4", "uint": "u4",
            "int32": "i4", "uint32": "u4",
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
        }
        if t not in m:
            raise ValueError(f"Unsupported PLY type: {t}")
        return np.dtype(m[t])

    with ply.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {ply}")
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element "):
                parts = s.split()
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                    props = []
                else:
                    in_vertex = False
            elif s.startswith("property ") and in_vertex:
                parts = s.split()
                if len(parts) >= 3 and parts[1] == "list":
                    raise RuntimeError(f"PLY vertex list properties not supported: {ply}")
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))  # (type, name)
            elif s == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise RuntimeError(f"PLY header missing format/vertex count: {ply}")

        names = [n for (_t, n) in props]
        for k in ("x", "y", "z"):
            if k not in names:
                raise RuntimeError(f"PLY missing vertex property '{k}': {ply}")

        has_rgb = all(k in names for k in ("red", "green", "blue"))
        has_rgb_alt = all(k in names for k in ("r", "g", "b"))

        if fmt != "ascii":
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = [(name, _dtype_from_type(typ).newbyteorder(endian)) for typ, name in props]
            data = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)
            if data.size == 0:
                raise RuntimeError(f"Empty PLY data: {ply}")

            pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

            rgb = None
            if has_rgb:
                rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
            elif has_rgb_alt:
                rgb = np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.uint8)

            return pts, rgb

        # ASCII
        name_to_idx = {name: i for i, (_typ, name) in enumerate(props)}
        x_i, y_i, z_i = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]
        r_i = g_i = b_i = None
        if has_rgb:
            r_i, g_i, b_i = name_to_idx["red"], name_to_idx["green"], name_to_idx["blue"]
        elif has_rgb_alt:
            r_i, g_i, b_i = name_to_idx["r"], name_to_idx["g"], name_to_idx["b"]

        xyz: List[List[float]] = []
        rgb_list: Optional[List[List[int]]] = [] if (r_i is not None) else None

        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.decode("ascii", errors="ignore").strip().split()
            if len(parts) <= max(x_i, y_i, z_i):
                continue
            xyz.append([float(parts[x_i]), float(parts[y_i]), float(parts[z_i])])
            if rgb_list is not None and len(parts) > max(r_i, g_i, b_i):
                rgb_list.append([
                    int(float(parts[r_i])),
                    int(float(parts[g_i])),
                    int(float(parts[b_i])),
                ])

        pts = np.asarray(xyz, dtype=np.float32)
        rgb = None
        if rgb_list is not None and len(rgb_list) == len(xyz):
            rgb = np.asarray(rgb_list, dtype=np.uint8)

        if pts.size == 0:
            raise RuntimeError(f"Laser scan is empty: {ply}")
        return pts, rgb


# -------------------------
# Mask loading
# -------------------------

def load_mask_png(path: Path, thr: float) -> np.ndarray:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV not available. Install: pip install opencv-python") from e

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return (m >= float(thr)).astype(np.uint8)


# -------------------------
# Projection visualization (B_Tcw only)
# -------------------------

def save_projection_image_b_tcw(
    *,
    out_path: Path,
    points_w: np.ndarray,
    rgb: Optional[np.ndarray],
    K: Intrinsics,
    T_cw: np.ndarray,
    sample_max: int = 200000,
    dot_radius: int = 2,
) -> Dict[str, int]:
    """
    Save a black image with projected points (using point cloud original rgb if present).

    Convention:
      T_cw = camera <- world
      p_c = R p_w + t
    """
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(K.h), int(K.w)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    p_c = (R @ points_w.T + t.reshape(3, 1)).T

    z = p_c[:, 2]
    in_front = z > 1e-6
    if not np.any(in_front):
        cv2.imwrite(str(out_path), img)
        return {"in_front": 0, "inside": 0, "unique_pixels": 0}

    p_c = p_c[in_front]
    z = p_c[:, 2]
    u = np.round((p_c[:, 0] * K.fx) / z + K.cx).astype(np.int32)
    v = np.round((p_c[:, 1] * K.fy) / z + K.cy).astype(np.int32)

    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    u = u[inside]
    v = v[inside]

    inside_n = int(u.size)
    if inside_n == 0:
        cv2.imwrite(str(out_path), img)
        return {"in_front": int(in_front.sum()), "inside": 0, "unique_pixels": 0}

    if inside_n > sample_max:
        idx = np.random.choice(inside_n, sample_max, replace=False)
        u = u[idx]
        v = v[idx]
        if rgb is not None:
            rgb2 = rgb[in_front][inside][idx]
        else:
            rgb2 = None
    else:
        rgb2 = rgb[in_front][inside] if rgb is not None else None

    if rgb2 is None:
        for uu, vv in zip(u, v):
            cv2.circle(img, (int(uu), int(vv)), int(dot_radius), (255, 255, 255), -1)
    else:
        # rgb2: RGB, opencv expects BGR
        for (uu, vv), (r, g, b) in zip(zip(u, v), rgb2):
            cv2.circle(img, (int(uu), int(vv)), int(dot_radius), (int(b), int(g), int(r)), -1)

    # unique pixel coverage (debug)
    uv = np.stack([u, v], axis=1)
    uniq = int(np.unique(uv, axis=0).shape[0])

    cv2.imwrite(str(out_path), img)
    return {"in_front": int(in_front.sum()), "inside": inside_n, "unique_pixels": uniq}


# -------------------------
# Core: world -> image -> indices (mask + optional depth consistency)
# -------------------------

def project_laser_to_mask_indices(
    points_w: np.ndarray,
    mask01: np.ndarray,
    depth_raw: np.ndarray,
    K: Intrinsics,
    T_cw: np.ndarray,
    vis_thres: float = 0.25,
    use_depth_filter: bool = True,
    depth_scale: float = 1000.0,   # depth_raw / depth_scale -> meters
) -> np.ndarray:
    """
    Project world points to image, keep points that fall inside mask.
    Pose convention: T_cw (camera <- world), so p_c = R p_w + t.

    If use_depth_filter:
      compare z (meters) with depth_raw/ depth_scale (meters)
      keep abs(depth - z) <= vis_thres * depth
    """
    if mask01.shape[:2] != depth_raw.shape[:2]:
        raise ValueError("mask and depth must have same HxW")
    h, w = mask01.shape[:2]
    if (K.h, K.w) != (h, w):
        raise ValueError(f"intrinsics size {K.w}x{K.h} != mask/depth {w}x{h}")

    # world -> camera (CORRECT for your dataset)
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    p_c = (R @ points_w.T + t.reshape(3, 1)).T  # (N,3)

    z = p_c[:, 2]
    in_front = z > 1e-6
    if not np.any(in_front):
        return np.zeros((0,), dtype=np.int64)

    p_c = p_c[in_front]
    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]

    u = np.round((x * K.fx) / z + K.cx).astype(np.int32)
    v = np.round((y * K.fy) / z + K.cy).astype(np.int32)

    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    if not np.any(inside):
        return np.zeros((0,), dtype=np.int64)

    u = u[inside]
    v = v[inside]
    z = z[inside]
    idx_world = np.nonzero(in_front)[0][inside]  # original indices into laser scan

    # mask hit
    mask_hit = mask01[v, u] > 0
    if not np.any(mask_hit):
        return np.zeros((0,), dtype=np.int64)

    u = u[mask_hit]
    v = v[mask_hit]
    z = z[mask_hit]
    idx_world = idx_world[mask_hit]

    if not use_depth_filter:
        return idx_world

    depth_here = depth_raw[v, u].astype(np.float32) / float(depth_scale)
    valid = depth_here > 1e-6
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)

    depth_here = depth_here[valid]
    z = z[valid]
    idx_world = idx_world[valid]

    occlusion = np.abs(depth_here - z) <= (vis_thres * depth_here)
    if not np.any(occlusion):
        return np.zeros((0,), dtype=np.int64)

    return idx_world[occlusion]


# -------------------------
# GT loading
# -------------------------

def load_gt_instances(
    annotations_json: Path,
    num_points: int,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    data = _safe_read_json(annotations_json)
    gt_masks: List[np.ndarray] = []
    gt_annot_ids: List[str] = []
    gt_labels: List[str] = []

    for a in data.get("annotations", []):
        annot_id = str(a.get("annot_id", "")).strip()
        label = str(a.get("label", "")).strip()
        idxs = a.get("indices", [])
        if not annot_id or not isinstance(idxs, list) or len(idxs) == 0:
            continue

        m = np.zeros((num_points,), dtype=bool)
        idxs_np = np.asarray(idxs, dtype=np.int64)
        idxs_np = idxs_np[(idxs_np >= 0) & (idxs_np < num_points)]
        m[idxs_np] = True

        gt_masks.append(m)
        gt_annot_ids.append(annot_id)
        gt_labels.append(label)

    return gt_masks, gt_annot_ids, gt_labels


def pred_indices_to_mask(indices: List[int], num_points: int) -> np.ndarray:
    m = np.zeros((num_points,), dtype=bool)
    if not indices:
        return m
    idx = np.asarray(indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < num_points)]
    m[idx] = True
    return m


# -------------------------
# COCO-style mAP over IoU thresholds
# -------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni)


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def ap_at_iou_threshold(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
    iou_thr: float,
) -> float:
    if len(gt_masks) == 0:
        return 0.0

    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32))
    pred_masks = [pred_masks[i] for i in order]
    pred_scores = [pred_scores[i] for i in order]

    matched = np.zeros((len(gt_masks),), dtype=bool)

    tp = np.zeros((len(pred_masks),), dtype=np.float32)
    fp = np.zeros((len(pred_masks),), dtype=np.float32)

    for i, pm in enumerate(pred_masks):
        best_j = -1
        best_iou = 0.0
        for j, gm in enumerate(gt_masks):
            if matched[j]:
                continue
            iou = mask_iou(pm, gm)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            matched[best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(len(gt_masks))

    return compute_ap_from_pr(precision, recall)


def coco_map(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
) -> Dict[str, float]:
    iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    aps = [ap_at_iou_threshold(gt_masks, pred_masks, pred_scores, t) for t in iou_thresholds]
    ap50 = ap_at_iou_threshold(gt_masks, pred_masks, pred_scores, 0.50)
    ap25 = ap_at_iou_threshold(gt_masks, pred_masks, pred_scores, 0.25)

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "AP50": float(ap50),
        "AP25": float(ap25),
    }


# -------------------------
# Core: pred.json -> indices (+ optional projection vis)
# -------------------------

def pred_json_to_indices(
    pred_json: Dict[str, Any],
    *,
    data_root: Path,
    scene_id: Union[int, str],
    masks_root: Optional[Path] = None,
    intr_dirname: str = "hires_wide_intrinsics",
    traj_name: str = "hires_poses.traj",
    depth_dirname: str = "hires_depth",
    mask_thr: float = 0.5,
    vis_thres: float = 0.25,
    use_depth_filter: bool = True,
    depth_scale: float = 1000.0,
    # visualization
    proj_vis_dir: Optional[Path] = None,
    proj_vis_limit_frames: int = 5,
    proj_vis_sample_max: int = 200000,
    proj_vis_dot_radius: int = 2,
) -> Dict[str, Dict[str, Any]]:
    scene_root = data_root / str(scene_id)
    seq_dir = find_first_sequence_dir(scene_root)

    intr_dir = find_intr_dir(seq_dir, intr_dirname)
    traj_path = find_traj_path(seq_dir, traj_name)
    T_list = read_traj_Twc(traj_path)  # 注意：函数名叫 Twc，但你已验证实际语义是 T_cw

    intr_files = sorted(intr_dir.iterdir(), key=lambda p: _natural_key(p.name))
    depth_dir = seq_dir / depth_dirname
    depth_files = list_depth_pngs(depth_dir)

    laser_pts, laser_rgb = _load_laser_scan_points_and_rgb(scene_root, scene_id)

    intr_cache: Dict[int, Intrinsics] = {}
    depth_cache: Dict[int, np.ndarray] = {}

    out: Dict[str, Dict[str, Any]] = {}
    saved_frames: set = set()

    for annot_id, entry in pred_json.items():
        frame_id = int(entry["frame"])
        mask_path = Path(entry["mask_png"])
        if masks_root is not None:
            mask_path = masks_root / mask_path.name

        if frame_id < 0 or frame_id >= len(intr_files) or frame_id >= len(depth_files) or frame_id >= len(T_list):
            continue

        if frame_id not in intr_cache:
            w, h, fx, fy, cx, cy = load_intrinsics_txt(intr_files[frame_id])
            intr_cache[frame_id] = Intrinsics(w, h, fx, fy, cx, cy)

        if frame_id not in depth_cache:
            depth_cache[frame_id] = load_depth_png(depth_files[frame_id])

        K = intr_cache[frame_id]
        depth_raw = depth_cache[frame_id]
        T_cw = T_list[frame_id]

        # projection visualization (once per frame)
        if proj_vis_dir is not None and frame_id not in saved_frames and len(saved_frames) < int(proj_vis_limit_frames):
            saved_frames.add(frame_id)
            out_img = proj_vis_dir / f"{scene_id}_frame{frame_id}_B_Tcw.png"
            stats = save_projection_image_b_tcw(
                out_path=out_img,
                points_w=laser_pts,
                rgb=laser_rgb,
                K=K,
                T_cw=T_cw,
                sample_max=int(proj_vis_sample_max),
                dot_radius=int(proj_vis_dot_radius),
            )
            cov = stats["unique_pixels"] / float(K.w * K.h)
            print(
                f"[PROJ-VIS] scene={scene_id} frame={frame_id} "
                f"in_front={stats['in_front']} inside={stats['inside']} "
                f"unique_pixels={stats['unique_pixels']} coverage={cov:.6f} "
                f"-> {out_img}"
            )

        mask01 = load_mask_png(mask_path, mask_thr)

        idxs = project_laser_to_mask_indices(
            laser_pts,
            mask01,
            depth_raw,
            K,
            T_cw,
            vis_thres=vis_thres,
            use_depth_filter=use_depth_filter,
            depth_scale=depth_scale,
        )

        uniq = sorted(set(int(i) for i in idxs.tolist()))
        out[str(annot_id)] = {
            "indices": uniq,
            "score": float(entry.get("score", 0.0)),
            "frame": int(frame_id),
            "mask_png": str(mask_path),
        }

    return out


def evaluate_indices(
    pred3d: Dict[str, Dict[str, Any]],
    annotations_json: Path,
    num_points: int,
) -> Dict[str, Any]:
    gt_masks, gt_annot_ids, _ = load_gt_instances(annotations_json, num_points=num_points)

    pred_masks: List[np.ndarray] = []
    pred_scores: List[float] = []
    for _annot_id, item in pred3d.items():
        idxs = item.get("indices", [])
        score = float(item.get("score", 0.0))
        m = pred_indices_to_mask(idxs, num_points=num_points)
        if m.sum() <= 0:
            continue
        pred_masks.append(m)
        pred_scores.append(score)

    metrics = coco_map(gt_masks=gt_masks, pred_masks=pred_masks, pred_scores=pred_scores)

    gt_id_to_mask = {aid: gt_masks[i] for i, aid in enumerate(gt_annot_ids)}
    aligned_ious = []
    for aid, gt_m in gt_id_to_mask.items():
        if aid not in pred3d:
            aligned_ious.append(0.0)
            continue
        pm = pred_indices_to_mask(pred3d[aid].get("indices", []), num_points=num_points)
        aligned_ious.append(mask_iou(pm, gt_m))
    miou_aligned = float(np.mean(aligned_ious)) if aligned_ious else 0.0

    return {
        **metrics,
        "mIoU_aligned_by_annot_id": miou_aligned,
        "gt_instances": int(len(gt_masks)),
        "pred_instances_nonempty": int(len(pred_masks)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", required=True, help="Path to pred.json / annot2bestmask.json")
    ap.add_argument("--data-root", required=True, help="Dataset root containing scene folders")
    ap.add_argument("--scene-id", required=True, help="Scene/visit id")
    ap.add_argument("--masks-root", default=None, help="Optional override root for mask_png files")

    ap.add_argument("--intr-dirname", default="hires_wide_intrinsics")
    ap.add_argument("--traj-name", default="hires_poses.traj")
    ap.add_argument("--depth-dirname", default="hires_depth")

    ap.add_argument("--mask-thr", type=float, default=0.5, help="Mask threshold for binarization")
    ap.add_argument("--vis-thres", type=float, default=0.25, help="Depth consistency threshold")
    ap.add_argument("--depth-scale", type=float, default=1000.0, help="Convert depth_raw to meters: depth/depth_scale")
    ap.add_argument("--no-depth-filter", action="store_true", help="Disable depth consistency filtering")

    ap.add_argument("--save-pred3d", default=None, help="Optional output path for pred3d.json")

    # projection visualization
    ap.add_argument("--proj-vis-dir", default=None, help="If set, save B_Tcw projection images here")
    ap.add_argument("--proj-vis-limit-frames", type=int, default=5, help="Save at most K frames")
    ap.add_argument("--proj-vis-sample-max", type=int, default=200000, help="Max points to draw per image")
    ap.add_argument("--proj-vis-dot-radius", type=int, default=2, help="Point radius in pixels (>=2 recommended)")

    args = ap.parse_args()

    pred_json = _safe_read_json(Path(args.pred_json))
    data_root = Path(args.data_root)
    scene_id = str(args.scene_id)

    _desc_json, ann_json = _resolve_scene_jsons(data_root, scene_id)

    scene_dir = data_root / scene_id
    laser_pts, _laser_rgb = _load_laser_scan_points_and_rgb(scene_dir, scene_id)
    num_points = int(laser_pts.shape[0])

    pred3d = pred_json_to_indices(
        pred_json,
        data_root=data_root,
        scene_id=scene_id,
        masks_root=Path(args.masks_root) if args.masks_root else None,
        intr_dirname=args.intr_dirname,
        traj_name=args.traj_name,
        depth_dirname=args.depth_dirname,
        mask_thr=float(args.mask_thr),
        vis_thres=float(args.vis_thres),
        use_depth_filter=(not args.no_depth_filter),
        depth_scale=float(args.depth_scale),
        proj_vis_dir=Path(args.proj_vis_dir) if args.proj_vis_dir else None,
        proj_vis_limit_frames=int(args.proj_vis_limit_frames),
        proj_vis_sample_max=int(args.proj_vis_sample_max),
        proj_vis_dot_radius=int(args.proj_vis_dot_radius),
    )

    metrics = evaluate_indices(pred3d, ann_json, num_points=num_points)

    if args.save_pred3d:
        Path(args.save_pred3d).write_text(
            json.dumps(pred3d, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
