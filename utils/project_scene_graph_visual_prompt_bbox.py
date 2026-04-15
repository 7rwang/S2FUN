#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
project_visual_prompt_only_text_vis80_nms_coverall_SOFTFAIL.py

ONLY-TEXT visual prompts by projecting each scene-graph node's POINT-SET to RGB frames,
computing per-node 2D bbox (for coverage + text placement), then selecting frames to cover nodes.

Key features:
- ONLY draw node ids on the RGB image (no bbox border, no bbox fill).
- Per-frame NMS to suppress highly-overlapping bboxes so numbers don't overlap.
- Optional ensure-cover stage to add frames for uncovered nodes.

SOFT-FAIL MODIFICATION:
- If some nodes have NO feasible projection in any considered frame (no_support),
  DO NOT raise; continue generating prompts for remaining nodes.
- If after selection some nodes remain uncovered, DO NOT raise; still render chosen frames.
- JSON report will include:
    - no_support_node_ids: nodes with zero feasible frames (hard-impossible under thresholds)
    - uncovered_node_ids: coverable nodes that selection/NMS didn't cover
  and "soft_fail": true.

Constraints:
- Node-level candidate must have >= --min_inliers projected in-image points.
- Node-level bbox visibility:
    vis_ratio = area(raw_bbox ∩ image)/area(raw_bbox) >= --min_bbox_vis_ratio (default 0.80)
- Frame-level eligible for phase-1:
    among candidates meeting min_inliers, fraction with vis_ratio>=threshold
    must be >= --frame_good_bbox_frac (default 0.80)

Outputs:
  out_dir/
    selected_frames.json
    frames/
      frame_0000.png
      ...

Example:
  python project_scene_graph_visual_prompt_bbox.py \
    --scene_graph_json /nas/.../scene_graph.json \
    --point_indices_json /nas/.../scene_graph_point_indices.json \
    --data_root /nas/qirui/scenefun3d/data \
    --out_dir /nas/.../visual_prompt_all_vis80_nms \
    --node_type all \
    --min_inliers 10 \
    --min_bbox_vis_ratio 0.80 \
    --frame_good_bbox_frac 0.80 \
    --suppress_iou 0.70 \
    --score_mode inliers_x_vis \
    --ensure_cover_all \
    --clear_out
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# -------------------------
# Utilities
# -------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def load_intrinsics_txt(p: Path) -> Tuple[int, int, float, float, float, float]:
    vals = p.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Intrinsics {p}: expect 6 values, got {len(vals)}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal length: fx={fx}, fy={fy}")
    return w, h, fx, fy, cx, cy


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).astype(np.float64)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def read_traj_Twc(traj_path: Path) -> List[np.ndarray]:
    T_list = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(f"Line {line_no}: expect 7 fields, got {len(parts)}")
            _, ax, ay, az, tx, ty, tz = map(float, parts)
            R = rodrigues(np.array([ax, ay, az], dtype=np.float64))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            T_list.append(T)
    if not T_list:
        raise ValueError(f"No valid poses in: {traj_path}")
    return T_list


def find_first_sequence_dir(scene_root: Path) -> Path:
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")
    seq_dirs = [p for p in scene_root.iterdir() if p.is_dir()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence dirs under: {scene_root}")
    numeric = [p for p in seq_dirs if re.fullmatch(r"\d+", p.name)]
    if numeric:
        return sorted(numeric, key=lambda p: natural_key(p.name))[0]
    return sorted(seq_dirs, key=lambda p: natural_key(p.name))[0]


def find_intr_dir(seq_dir: Path) -> Path:
    p = seq_dir / "hires_wide_intrinsics"
    if not p.exists():
        raise FileNotFoundError(f"Intrinsics dir not found: {p}")
    return p


def find_traj_path(seq_dir: Path) -> Path:
    p = seq_dir / "hires_poses.traj"
    if not p.exists():
        raise FileNotFoundError(f"Trajectory not found: {p}")
    return p


def list_files_sorted(d: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))


def find_rgb_dir_auto(seq_dir: Path) -> Path:
    candidates = [
        "hires_wide", "hires_wide_color", "hires_wide_rgb",
        "hires", "rgb", "color", "images", "image", "frames"
    ]
    for c in candidates:
        p = seq_dir / c
        if p.exists() and p.is_dir():
            imgs = list_files_sorted(p, (".png", ".jpg", ".jpeg"))
            if imgs:
                return p
    for p in sorted([x for x in seq_dir.iterdir() if x.is_dir()], key=lambda x: natural_key(x.name)):
        imgs = list_files_sorted(p, (".png", ".jpg", ".jpeg"))
        if imgs:
            return p
    raise FileNotFoundError(f"Cannot find an RGB image directory under: {seq_dir}")


def load_image(p: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV not available (cv2 import failed).")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to read image: {p}")
    return im


# -------------------------
# Point cloud loading
# -------------------------

def load_pcd_points(pcd_path: Path) -> np.ndarray:
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")
    suf = pcd_path.suffix.lower()
    if suf == ".npy":
        pts = np.load(str(pcd_path)).astype(np.float32)
    elif suf == ".npz":
        z = np.load(str(pcd_path))
        key = next((k for k in ["points", "xyz", "pcd"] if k in z), None)
        if key is None:
            raise ValueError(f"NPZ has no point key. Available: {list(z.keys())[:10]}")
        pts = np.asarray(z[key], dtype=np.float32)
    elif suf in [".ply", ".pcd"]:
        try:
            import open3d as o3d
        except ImportError as e:
            raise RuntimeError("Need open3d for .ply/.pcd files.") from e
        pts = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported format: {pcd_path}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Point cloud must be (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {pcd_path}")
    return pts


# -------------------------
# Projection + bbox visibility
# -------------------------

def project_cam_to_pixel(xyz_cam: np.ndarray, fx, fy, cx, cy):
    z = xyz_cam[:, 2].astype(np.float64)
    u = (xyz_cam[:, 0].astype(np.float64) * fx / z) + cx
    v = (xyz_cam[:, 1].astype(np.float64) * fy / z) + cy
    return u, v


def raw_bbox_from_uv(u: np.ndarray, v: np.ndarray, margin: int = 0) -> Tuple[float, float, float, float]:
    umin = float(np.min(u)) - margin
    umax = float(np.max(u)) + margin
    vmin = float(np.min(v)) - margin
    vmax = float(np.max(v)) + margin
    return (umin, vmin, umax, vmax)


def intersect_bbox_with_image(raw: Tuple[float, float, float, float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    umin, vmin, umax, vmax = raw
    x1 = max(0, min(w - 1, int(math.floor(umin))))
    y1 = max(0, min(h - 1, int(math.floor(vmin))))
    x2 = max(0, min(w - 1, int(math.ceil(umax))))
    y2 = max(0, min(h - 1, int(math.ceil(vmax))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_area(x1, y1, x2, y2) -> float:
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def bbox_visible_ratio(raw: Tuple[float, float, float, float], w: int, h: int) -> float:
    umin, vmin, umax, vmax = raw
    raw_area = float(max(0.0, umax - umin) * max(0.0, vmax - vmin))
    if raw_area <= 1e-12:
        return 0.0
    inter = intersect_bbox_with_image(raw, w, h)
    if inter is None:
        return 0.0
    x1, y1, x2, y2 = inter
    inter_area = bbox_area(x1, y1, x2, y2)
    return float(np.clip(inter_area / raw_area, 0.0, 1.0))


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = bbox_area(ix1, iy1, ix2, iy2)
    if inter <= 0:
        return 0.0
    ua = bbox_area(ax1, ay1, ax2, ay2) + bbox_area(bx1, by1, bx2, by2) - inter
    return float(inter / ua) if ua > 0 else 0.0


# -------------------------
# Rendering: ONLY TEXT
# -------------------------

def blend_overlay(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return base_bgr
    if alpha >= 1:
        return overlay_bgr
    return cv2.addWeighted(overlay_bgr, alpha, base_bgr, 1.0 - alpha, 0.0)


def draw_only_text(
    img_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int, int]],
    text_scale: float,
    text_thickness: int,
    text_alpha: float,
    text_pos: str = "center",
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 required for drawing.")
    out = img_bgr.copy()
    if text_alpha <= 0:
        return out

    overlay = out.copy()
    H, W = out.shape[:2]

    for nid, x1, y1, x2, y2 in boxes:
        txt = str(nid)

        if text_pos == "topleft":
            px, py = x1 + 4, y1 + 18
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, float(text_scale), int(text_thickness))
            px, py = int(cx - tw // 2), int(cy + th // 2)

        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        cv2.putText(overlay, txt, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, float(text_scale),
                    (0, 0, 0), int(text_thickness) + 2, cv2.LINE_AA)
        cv2.putText(overlay, txt, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, float(text_scale),
                    (255, 255, 255), int(text_thickness), cv2.LINE_AA)

    return blend_overlay(out, overlay, float(text_alpha))


# -------------------------
# Greedy set cover + ensure cover
# -------------------------

def greedy_min_frames_to_cover(cover_map: Dict[int, set], target_nodes: set, max_frames: int = -1) -> List[int]:
    uncovered = set(target_nodes)
    chosen: List[int] = []
    frames = {k: (v & target_nodes) for k, v in cover_map.items() if v}

    while uncovered:
        best_f = None
        best_gain = 0
        for f, s in frames.items():
            gain = len(s & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_f = f
        if best_f is None or best_gain == 0:
            break
        chosen.append(best_f)
        uncovered -= frames[best_f]
        if max_frames > 0 and len(chosen) >= max_frames:
            break
    return chosen


# -------------------------
# NMS suppression per frame
# -------------------------

def suppress_overlaps(
    candidates: List[dict],
    suppress_iou: float,
) -> List[dict]:
    """
    candidates: list of dict with fields:
      - nid (int)
      - bbox (x1,y1,x2,y2) int (already clamped to image)
      - score (float)
    Returns kept list after NMS.
    """
    if suppress_iou <= 0:
        return candidates

    cand = sorted(candidates, key=lambda x: (-float(x["score"]), x["nid"]))
    kept: List[dict] = []
    for c in cand:
        ok = True
        for k in kept:
            if iou_xyxy(tuple(c["bbox"]), tuple(k["bbox"])) >= suppress_iou:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ONLY-TEXT visual prompts with bbox visibility constraints + NMS + cover-all (SOFT-FAIL)."
    )
    ap.add_argument("--scene_graph_json", type=str, required=True)
    ap.add_argument("--point_indices_json", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--pcd_path", type=str, default="")
    ap.add_argument("--scene_id", type=int, default=-1)
    ap.add_argument("--sequence_id", type=str, default="")
    ap.add_argument("--rgb_dir", type=str, default="")

    ap.add_argument("--node_type", type=str, default="all", choices=["all", "object", "affordance"])

    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=-1)

    ap.add_argument("--sample_points_per_node", type=int, default=2000,
                    help="0=all points (slow but stable)")
    ap.add_argument("--min_inliers", type=int, default=10)
    ap.add_argument("--bbox_margin", type=int, default=2)

    ap.add_argument("--min_bbox_vis_ratio", type=float, default=0.80)
    ap.add_argument("--frame_good_bbox_frac", type=float, default=0.80)

    ap.add_argument("--suppress_iou", type=float, default=0.70,
                    help="If bbox IoU >= this threshold, keep only one node id in that frame.")
    ap.add_argument("--score_mode", type=str, default="inliers_x_vis",
                    choices=["inliers", "inliers_x_vis", "area_x_vis"],
                    help="Score used for NMS priority.")

    ap.add_argument("--ensure_cover_all", action="store_true",
                    help="Force-add frames so every coverable node appears at least once (if feasible).")
    ap.add_argument("--max_extra_frames", type=int, default=-1,
                    help="Cap extra frames added in ensure_cover_all (-1 = no cap).")

    # text
    ap.add_argument("--text_scale", type=float, default=1.2)
    ap.add_argument("--text_thickness", type=int, default=1)
    ap.add_argument("--text_alpha", type=float, default=0.95)
    ap.add_argument("--text_pos", type=str, default="center", choices=["center", "topleft"])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--clear_out", action="store_true")

    args = ap.parse_args()

    if cv2 is None:
        raise RuntimeError("cv2 not available in your env.")

    sg_path = Path(args.scene_graph_json)
    pi_path = Path(args.point_indices_json)
    data_root = Path(args.data_root)

    out_dir = Path(args.out_dir)
    out_frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_out:
        if out_frames_dir.exists():
            for p in out_frames_dir.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                except Exception:
                    pass
        out_frames_dir.mkdir(parents=True, exist_ok=True)

    sg = json.loads(sg_path.read_text(encoding="utf-8"))
    pi = json.loads(pi_path.read_text(encoding="utf-8"))

    # id->type from scene_graph.json
    id_to_type: Dict[int, str] = {}
    for n in sg.get("nodes", []):
        try:
            nid = int(n.get("id"))
            id_to_type[nid] = str(n.get("type", "")).strip().lower()
        except Exception:
            continue

    scene_id = args.scene_id if args.scene_id != -1 else int(sg.get("scene_id", -1))
    if scene_id == -1:
        raise ValueError("scene_id missing.")
    scene_root = data_root / str(scene_id)
    if not scene_root.exists():
        raise FileNotFoundError(f"scene_root not found: {scene_root}")

    seq_id = args.sequence_id.strip() if args.sequence_id.strip() else str(sg.get("sequence_id", "")).strip()
    if seq_id:
        seq_dir = scene_root / seq_id
        if not seq_dir.exists():
            seq_dir = find_first_sequence_dir(scene_root)
    else:
        seq_dir = find_first_sequence_dir(scene_root)

    # PCD path
    if args.pcd_path.strip():
        pcd_path = Path(args.pcd_path)
    else:
        pcd_path_str = str(sg.get("pcd_path", "")).strip()
        if not pcd_path_str:
            raise ValueError("pcd_path missing in scene_graph.json; pass --pcd_path")
        pcd_path = Path(pcd_path_str)
        if not pcd_path.is_absolute():
            pcd_path = (scene_root / pcd_path)

    print(">>> RUNNING: ONLY_TEXT + vis-constraints + NMS + (optional) ensure-cover (SOFT-FAIL) <<<")
    print(f"[PCD] {pcd_path}")
    world_pts = load_pcd_points(pcd_path)
    print(f"[PCD] points={len(world_pts):,}")

    # node -> indices (filter type)
    want = str(args.node_type).strip().lower()
    node_rows = pi.get("node_point_indices", [])
    if not node_rows:
        raise ValueError("point_indices_json has no node_point_indices")

    node_to_indices: Dict[int, np.ndarray] = {}
    for r in node_rows:
        nid = int(r["id"])
        ntype = id_to_type.get(nid, "").lower()
        if want != "all" and ntype != want:
            continue
        idx = np.asarray(r["point_indices"], dtype=np.int64)
        if idx.size > 0:
            node_to_indices[nid] = idx

    all_nodes = set(node_to_indices.keys())
    if not all_nodes:
        raise ValueError(f"No nodes after filtering node_type={want}")

    # intr + traj + rgb
    intr_dir = find_intr_dir(seq_dir)
    intr_files = sorted([p for p in intr_dir.iterdir() if p.is_file()], key=lambda p: natural_key(p.name))
    traj_path = find_traj_path(seq_dir)
    T_cw_list = read_traj_Twc(traj_path)

    if args.rgb_dir.strip():
        rgb_dir = Path(args.rgb_dir)
    else:
        rgb_dir = find_rgb_dir_auto(seq_dir)
    rgb_files = list_files_sorted(rgb_dir, (".png", ".jpg", ".jpeg"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb images under: {rgb_dir}")

    n_frames = min(len(rgb_files), len(intr_files), len(T_cw_list))
    stride = max(1, int(args.frame_stride))
    frame_ids = list(range(0, n_frames, stride))

    rng = np.random.default_rng(int(args.seed))

    print(f"[Info] scene_id={scene_id} seq_dir={seq_dir}")
    print(f"[Info] rgb_dir={rgb_dir}")
    print(f"[Info] aligned_frames={n_frames}, considered={len(frame_ids)} (stride={stride})")
    print(f"[Info] node_type={want}, nodes={len(all_nodes)}")
    print(f"[Param] min_inliers={args.min_inliers}, min_bbox_vis_ratio={args.min_bbox_vis_ratio}, "
          f"frame_good_bbox_frac={args.frame_good_bbox_frac}, suppress_iou={args.suppress_iou}, score_mode={args.score_mode}")

    # For each node, store feasible frames (node-level constraints)
    # node_support[nid] = list of (frame_id, score, bbox_xyxy_int, inliers, vis_ratio)
    node_support: Dict[int, List[Tuple[int, float, Tuple[int, int, int, int], int, float]]] = {nid: [] for nid in all_nodes}

    # For phase-1 cover:
    eligible_cover_map: Dict[int, set] = {}
    eligible_frame_boxes: Dict[int, List[Tuple[int, int, int, int, int]]] = {}  # (nid,x1,y1,x2,y2)

    # Any-frame cover/boxes (used in ensure-cover stage)
    any_frame_boxes: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    any_frame_cover: Dict[int, set] = {}

    frame_debug: Dict[int, dict] = {}

    for it, fi in enumerate(frame_ids):
        w, h, fx, fy, cx, cy = load_intrinsics_txt(intr_files[fi])
        T_cw = T_cw_list[fi]

        cand_total = 0
        good_total = 0
        candidates: List[dict] = []

        for nid, idx_all in node_to_indices.items():
            if idx_all.size == 0:
                continue

            spn = int(args.sample_points_per_node)
            if spn > 0 and idx_all.size > spn:
                sel = rng.choice(idx_all, size=spn, replace=False)
            else:
                sel = idx_all

            pts_world = world_pts[sel].astype(np.float64)
            pts_cam = apply_T(T_cw, pts_world)

            z = pts_cam[:, 2]
            valid = np.isfinite(pts_cam).all(axis=1) & np.isfinite(z) & (z > 0)
            if not np.any(valid):
                continue

            u, v = project_cam_to_pixel(pts_cam[valid], fx, fy, cx, cy)
            u_i = np.round(u).astype(np.int32)
            v_i = np.round(v).astype(np.int32)

            inside = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
            if not np.any(inside):
                continue

            u_in = u_i[inside]
            v_in = v_i[inside]
            inliers = int(u_in.size)
            if inliers < int(args.min_inliers):
                continue

            cand_total += 1

            raw_bb = raw_bbox_from_uv(u_in.astype(np.float64), v_in.astype(np.float64), margin=int(args.bbox_margin))
            vis = float(bbox_visible_ratio(raw_bb, w, h))

            good = (vis >= float(args.min_bbox_vis_ratio))
            if good:
                good_total += 1
            else:
                continue  # not feasible for this node

            inter = intersect_bbox_with_image(raw_bb, w, h)
            if inter is None:
                continue
            x1, y1, x2, y2 = inter

            area = bbox_area(x1, y1, x2, y2)
            if args.score_mode == "inliers":
                score = float(inliers)
            elif args.score_mode == "inliers_x_vis":
                score = float(inliers) * float(vis)
            else:  # area_x_vis
                score = float(area) * float(vis)

            node_support[int(nid)].append((int(fi), float(score), (int(x1), int(y1), int(x2), int(y2)), int(inliers), float(vis)))

            candidates.append({
                "nid": int(nid),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "inliers": int(inliers),
                "vis": float(vis),
                "score": float(score),
            })

        good_frac = (good_total / cand_total) if cand_total > 0 else 0.0
        frame_ok = (cand_total > 0) and (good_frac >= float(args.frame_good_bbox_frac))

        kept = suppress_overlaps(candidates, suppress_iou=float(args.suppress_iou))
        kept_nodes = set([c["nid"] for c in kept])
        boxes_kept = [(c["nid"], *c["bbox"]) for c in kept]

        if kept_nodes:
            any_frame_cover[fi] = kept_nodes
            any_frame_boxes[fi] = boxes_kept

        if frame_ok and kept_nodes:
            eligible_cover_map[fi] = kept_nodes
            eligible_frame_boxes[fi] = boxes_kept

        frame_debug[fi] = {
            "cand_total_min_inliers": int(cand_total),
            "good_total_vis_ok": int(good_total),
            "good_frac": float(good_frac),
            "frame_ok": bool(frame_ok),
            "kept_after_nms": int(len(kept_nodes)),
        }

        if args.log_every > 0 and (it % int(args.log_every) == 0 or it == len(frame_ids) - 1):
            print(f"  frame={fi:04d} cand={cand_total:3d} good={good_total:3d} "
                  f"good_frac={good_frac:.3f} frame_ok={int(frame_ok)} kept_nms={len(kept_nodes):3d}")

    # -------------------------
    # SOFT-FAIL: handle no_support without raising
    # -------------------------
    no_support = sorted([nid for nid, lst in node_support.items() if not lst])
    coverable_nodes = set(all_nodes) - set(no_support)

    if no_support:
        print(f"[WARN][SOFT-FAIL] nodes with no feasible frame under thresholds: {len(no_support)} -> {no_support}")

    # If nothing coverable, still output a JSON and exit normally
    if not coverable_nodes:
        report = {
            "scene_id": int(scene_id),
            "sequence_dir": str(seq_dir),
            "rgb_dir": str(rgb_dir),
            "pcd_path": str(pcd_path),
            "node_type": want,
            "n_nodes_total": int(len(all_nodes)),
            "n_nodes_coverable": 0,
            "no_support_node_ids": no_support,
            "uncovered_node_ids": [],
            "chosen_frames": [],
            "soft_fail": True,
            "message": "No coverable nodes exist under current thresholds. Nothing rendered.",
            "params": {
                "min_inliers": int(args.min_inliers),
                "min_bbox_vis_ratio": float(args.min_bbox_vis_ratio),
                "frame_good_bbox_frac": float(args.frame_good_bbox_frac),
                "suppress_iou": float(args.suppress_iou),
                "score_mode": str(args.score_mode),
                "frame_stride": int(args.frame_stride),
                "sample_points_per_node": int(args.sample_points_per_node),
                "ensure_cover_all": bool(args.ensure_cover_all),
            },
        }
        out_json = out_dir / "selected_frames.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK][SOFT-FAIL] wrote: {out_json}")
        return

    if not eligible_cover_map:
        # SOFT-FAIL: No eligible frames for phase-1, but we can still attempt cover using any_frame_cover
        print("[WARN][SOFT-FAIL] No eligible frames after frame-level constraint. Falling back to any-frame cover for selection.")
        cover_map_for_selection = any_frame_cover
        frame_boxes_for_render_primary = any_frame_boxes
        selection_source = "any_only"
    else:
        cover_map_for_selection = eligible_cover_map
        frame_boxes_for_render_primary = eligible_frame_boxes
        selection_source = "eligible_first"

    # Phase-1: greedy cover over coverable_nodes
    chosen = greedy_min_frames_to_cover(cover_map_for_selection, coverable_nodes, max_frames=int(args.max_frames))

    covered_all = set()
    for f in chosen:
        covered_all |= cover_map_for_selection.get(f, set())

    uncovered = set(coverable_nodes) - covered_all

    # Phase-2: ensure cover all coverable nodes (optional)
    extra_added = []
    if args.ensure_cover_all and uncovered:
        uncovered_list = sorted(list(uncovered))
        for nid in uncovered_list:
            lst = sorted(node_support[nid], key=lambda x: (-x[1], x[0]))
            picked_frame = None

            # Prefer frames where the node survives NMS (so it will be drawn)
            for fi, score, bb, inliers, vis in lst:
                if fi in any_frame_cover and nid in any_frame_cover[fi]:
                    picked_frame = fi
                    break

            # If NMS always suppresses it, just pick its best frame anyway (it may still be absent in render if suppressed)
            if picked_frame is None:
                picked_frame = lst[0][0]

            if picked_frame is not None and picked_frame not in chosen:
                chosen.append(picked_frame)
                extra_added.append({"node_id": int(nid), "frame_index": int(picked_frame)})

                if int(args.max_extra_frames) > 0 and len(extra_added) >= int(args.max_extra_frames):
                    break

        # Recompute coverage using union of cover maps
        chosen = sorted(set(chosen))
        covered_all = set()
        for f in chosen:
            if f in eligible_cover_map:
                covered_all |= eligible_cover_map[f]
            elif f in any_frame_cover:
                covered_all |= any_frame_cover[f]
        uncovered = set(coverable_nodes) - covered_all

    chosen = sorted(set(chosen))
    uncovered_sorted = sorted(list(uncovered))

    if uncovered_sorted:
        print(f"[WARN][SOFT-FAIL] still uncovered coverable nodes after selection: {len(uncovered_sorted)} -> {uncovered_sorted}")

    # Render chosen frames (ONLY TEXT). Prefer eligible boxes if frame is eligible; else any-frame boxes.
    saved = []
    for f in chosen:
        img = load_image(rgb_files[f])

        if f in eligible_frame_boxes:
            boxes = eligible_frame_boxes[f]
            cover = eligible_cover_map.get(f, set())
            src = "eligible"
        else:
            boxes = any_frame_boxes.get(f, [])
            cover = any_frame_cover.get(f, set())
            src = "any"

        boxes = sorted(boxes, key=lambda t: (t[0], t[2], t[1]))

        img2 = draw_only_text(
            img_bgr=img,
            boxes=boxes,
            text_scale=float(args.text_scale),
            text_thickness=int(args.text_thickness),
            text_alpha=float(args.text_alpha),
            text_pos=str(args.text_pos),
        )

        out_img = out_frames_dir / f"frame_{f:04d}.png"
        cv2.imwrite(str(out_img), img2)

        saved.append({
            "frame_index": int(f),
            "rgb_path": str(rgb_files[f]),
            "covers_node_ids": sorted(list(cover & coverable_nodes)),
            "num_texts": int(len(boxes)),
            "frame_debug": frame_debug.get(f, {}),
            "source": src,
            "out_image": str(out_img),
        })

    report = {
        "scene_id": int(scene_id),
        "sequence_dir": str(seq_dir),
        "scene_graph_json": str(sg_path),
        "point_indices_json": str(pi_path),
        "pcd_path": str(pcd_path),
        "rgb_dir": str(rgb_dir),

        "node_type": want,
        "frame_stride": int(args.frame_stride),
        "n_frames_total_aligned": int(n_frames),
        "n_frames_considered": int(len(frame_ids)),
        "n_frames_eligible_phase1": int(len(eligible_cover_map)),

        "n_nodes_total": int(len(all_nodes)),
        "n_nodes_coverable": int(len(coverable_nodes)),
        "no_support_node_ids": no_support,
        "uncovered_node_ids": uncovered_sorted,

        "chosen_frames": saved,
        "covered_node_ids": sorted(list(covered_all & coverable_nodes)),

        "extra_added": extra_added,

        "selection_source": selection_source,
        "soft_fail": True,

        "params": {
            "sample_points_per_node": int(args.sample_points_per_node),
            "min_inliers": int(args.min_inliers),
            "bbox_margin": int(args.bbox_margin),

            "min_bbox_vis_ratio": float(args.min_bbox_vis_ratio),
            "frame_good_bbox_frac": float(args.frame_good_bbox_frac),

            "suppress_iou": float(args.suppress_iou),
            "score_mode": str(args.score_mode),

            "ensure_cover_all": bool(args.ensure_cover_all),
            "max_extra_frames": int(args.max_extra_frames),

            "text_scale": float(args.text_scale),
            "text_thickness": int(args.text_thickness),
            "text_alpha": float(args.text_alpha),
            "text_pos": str(args.text_pos),

            "seed": int(args.seed),
            "max_frames": int(args.max_frames),
            "clear_out": bool(args.clear_out),
        },
        "note": (
            "SOFT-FAIL mode: nodes with no feasible projection do not abort the run; "
            "they are reported in no_support_node_ids. "
            "If some coverable nodes are still uncovered (likely due to NMS/selection), "
            "they are reported in uncovered_node_ids."
        ),
    }

    out_json = out_dir / "selected_frames.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote: {out_json}")
    print(f"[OK] frames: {out_frames_dir}")
    print(f"[OK] chosen_frames={len(chosen)}  coverable_nodes={len(coverable_nodes)}  "
          f"no_support={len(no_support)}  uncovered={len(uncovered_sorted)}")


if __name__ == "__main__":
    main()