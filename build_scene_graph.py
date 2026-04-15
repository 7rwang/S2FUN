#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_scene_graph_vote_voxel_opt.py  ── OPTIMIZED VERSION + MODIFIED

Key optimizations vs. original:
  [OPT-1] Frame-level projection cache  — project world_pts → pixels ONCE per frame,
           reused by every mask in that frame (was: once per mask).
  [OPT-2] Vectorized add_obs            — replace Python loops with numpy unique+add.
  [OPT-3] Vectorized vote filter        — replace per-point Python loop with numpy ops
           (including Wilson lower-bound computed in bulk).
  [OPT-4] Vectorized voxel-cap          — use the existing fast path everywhere; also
           applied to the cached projection arrays before mask lookup.
  [OPT-5] Faster merge inverted-index   — numpy sort + split instead of Python dict loop.
  [OPT-6] SemanticGroup stores arrays   — vis_count / fg_count kept as numpy int32 arrays
           indexed by a compact local id; converted to dict only when needed for vote.
  [OPT-7] depth-consistency per frame   — pre-compute median/MAD per frame (not per mask)
           when all masks share the same depth image.

Additional modifications in this version:
  [MOD-1] Object-only 3D bbox merge:
           Merge object nodes if their 3D AABBs have large overlap or strong containment.
           This merge is restricted to object level only.
  [MOD-2] Affordance-priority ownership:
           If point in (object ∩ affordance), keep only in affordance.

python build_scene_graph.py \
  --scene_id 421254 \
  --masks_root /nas/qirui/sam3/scenefun3d_ex/refined_masks_3d/421254/masks \
  --data_root /nas/qirui/scenefun3d/val \
  --depth_scale 1000 \
  --out_json /nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/0.02/421254/scene_graph.json \
  --save_ply \
  --use_depth_consistency \
  --min_vis 5 \
  --min_vis_frac 0 \
  --min_fg 2 \
  --aff_min_vis 5 \
  --aff_min_fg 2 \
  --aff_min_fg_ratio 0.05 \
  --aff_min_vis_frac 0.02 \
  --min_fg_ratio 0.3 \
  --use_dbscan_position \
  --dbscan_eps 1 \
  --aff_dbscan_eps 0.03 \
  --dbscan_min_samples 10 \
  --voxel_size 0 \
  --merge_nodes \
  --merge_min_common 50 \
  --merge_min_ratio 0.3 \
  --merge_object_bbox \
  --bbox_iou_thr 0.25 \
  --bbox_contain_thr 0.80

"""

import argparse
import json
import math
import os
import re
import shutil
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).astype(np.float64)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def validate_transform(T: np.ndarray, name: str = "Transform") -> None:
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(det) < 0.01:
        raise ValueError(f"{name} has near-zero determinant: {det}")
    if np.max(np.abs(R.T @ R - np.eye(3))) > 0.1:
        raise ValueError(f"{name} rotation not orthogonal")


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
            try:
                _, ax, ay, az, tx, ty, tz = map(float, parts)
                R = rodrigues(np.array([ax, ay, az]))
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                validate_transform(T, f"Trajectory line {line_no}")
                T_list.append(T)
            except Exception as e:
                raise ValueError(f"Line {line_no}: {e}")
    if not T_list:
        raise ValueError(f"No valid poses in: {traj_path}")
    return T_list


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


def load_mask(mask_path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV not available.")
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def subtract_union_mask(mask01: np.ndarray, union_bool: Optional[np.ndarray]) -> np.ndarray:
    if union_bool is None:
        return mask01
    if union_bool.shape != mask01.shape:
        raise ValueError(f"Union mask shape {union_bool.shape} != mask shape {mask01.shape}")
    return (mask01.astype(bool) & (~union_bool)).astype(np.uint8)


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def list_frame_dirs(masks_root: Path) -> List[int]:
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")
    sub = [p for p in masks_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if not sub:
        return [0]
    return sorted([int(p.name) for p in sub])


def get_frame_dir(masks_root: Path, idx: int) -> Path:
    p = masks_root / str(idx)
    return p if p.is_dir() else masks_root


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    synonyms = {"drawers": "drawer", "handles": "handle", "doors": "door", "cabinets": "cabinet"}
    return synonyms.get(name, name)


def load_name_map_json(p: Optional[Path]) -> Dict[str, str]:
    if p is None:
        return {}
    if not p.exists():
        raise FileNotFoundError(f"name_map_json not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
        for k, v in obj["map"].items():
            if k and v:
                mapping[normalize_name(str(k))] = normalize_name(str(v))
    if isinstance(obj, dict) and "groups" in obj and isinstance(obj["groups"], list):
        for g in obj["groups"]:
            if isinstance(g, list) and g:
                canon = normalize_name(str(g[0]))
                for item in g:
                    mapping[normalize_name(str(item))] = canon
    if isinstance(obj, dict) and "map" not in obj and "groups" not in obj:
        for k, v in obj.items():
            mapping[normalize_name(str(k))] = normalize_name(str(v))

    def resolve(x):
        seen, cur = set(), x
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        return cur

    return {k: resolve(v) for k, v in mapping.items()}


def canon_name(name: str, name_map: Dict[str, str]) -> str:
    return name_map.get(normalize_name(name), normalize_name(name))


def parse_mask_type_name_inst(stem: str) -> Tuple[str, str, str]:
    """
    Support both original names and refined names.

    Original examples:
      CTX__door__000__area1234
      INT__door_handle__000__area3475

    Refined examples:
      instance_0001__CTX__door__000__area1234
      instance_0007__INT__door_handle__000__area3475

    Returns:
      typ: "object" or "affordance"
      name: normalized semantic name, e.g. "door handle"
      inst_id: local mask id parsed from filename (NOT the instance_xxxx prefix)
    """
    s = stem.strip()

    # 1) strip refined prefix like "instance_0001__"
    #    allow both lower/upper case just in case
    s = re.sub(r"^instance_\d+__", "", s, flags=re.IGNORECASE)

    # 2) determine type from the remaining stem
    if s.startswith("CTX"):
        typ, prefix = "object", "CTX"
    elif s.startswith("INT"):
        typ, prefix = "affordance", "INT"
    else:
        typ, prefix = "object", None

    # 3) split by "__" first; if not present, fall back to "_"
    parts = [p for p in (s.split("__") if "__" in s else s.split("_")) if p]

    # 4) remove leading CTX / INT token if present
    if prefix and parts and parts[0] == prefix:
        parts = parts[1:]
    elif prefix and parts and parts[0].startswith(prefix):
        parts[0] = parts[0].replace(prefix, "", 1)

    # 5) parse local mask id, stop before area1234
    inst_id = "000"
    name_tokens = []
    for p in parts:
        if re.fullmatch(r"\d+", p):
            inst_id = p.zfill(3)
            break
        if re.fullmatch(r"area\d+", p, flags=re.IGNORECASE):
            break
        name_tokens.append(p)

    name = normalize_name(" ".join(name_tokens).strip() or "unknown")
    return typ, name, inst_id

# ─────────────────────────────────────────────
# Depth helpers
# ─────────────────────────────────────────────

def load_depth(depth_path: Path) -> np.ndarray:
    suf = depth_path.suffix.lower()
    if suf == ".npy":
        d = np.load(str(depth_path))
        return (d[..., 0] if d.ndim == 3 else d).astype(np.float32)
    if cv2 is not None:
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read depth: {depth_path}")
        return (d[..., 0] if d.ndim == 3 else d).astype(np.float32)
    from PIL import Image
    d = np.array(Image.open(depth_path))
    return (d[..., 0] if d.ndim == 3 else d).astype(np.float32)


def find_depth_dir_auto(seq_dir: Path) -> Optional[Path]:
    for cand in ["hires_wide_depth", "hires_depth", "depth", "depths"]:
        p = seq_dir / cand
        if p.exists() and p.is_dir():
            return p
    return None


def list_depth_files(depth_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy")
    files = [p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))


# ─────────────────────────────────────────────
# Wilson lower bound
# ─────────────────────────────────────────────

def wilson_lower_bound(k: int, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    k = max(0, min(k, n))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    inside = max(0.0, phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return float(max(0.0, min(1.0, (center - z * math.sqrt(inside)) / denom)))


def wilson_lower_bound_vec(k: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    n = np.maximum(n, 1).astype(np.float64)
    k = np.clip(k, 0, n).astype(np.float64)
    z2 = z * z
    phat = k / n
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    inside = np.maximum(phat * (1.0 - phat) / n + z2 / (4.0 * n * n), 0.0)
    lb = (center - z * np.sqrt(inside)) / denom
    return np.clip(lb, 0.0, 1.0)


# ─────────────────────────────────────────────
# Voxel cap
# ─────────────────────────────────────────────

def voxel_cap_indices_fast(indices: np.ndarray, world_pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if indices.size == 0 or voxel_size <= 0.0:
        return indices
    pts = world_pts[indices]
    vox = np.floor(pts / voxel_size).astype(np.int64)
    offset = vox.min(axis=0)
    shifted = vox - offset
    maxv = shifted.max(axis=0) + 1
    total = int(maxv[0]) * int(maxv[1]) * int(maxv[2])
    if total > 2**50:
        seen: dict = {}
        keep = np.zeros(len(indices), dtype=bool)
        for li, key in enumerate(map(tuple, vox.tolist())):
            if key not in seen:
                seen[key] = li
                keep[li] = True
        return indices[keep]
    vox_id = (shifted[:, 0] + maxv[0] * (shifted[:, 1] + maxv[1] * shifted[:, 2]))
    order = np.argsort(vox_id, kind="stable")
    _, first_in_sorted = np.unique(vox_id[order], return_index=True)
    local_positions = np.sort(order[first_in_sorted])
    return indices[local_positions]


# ─────────────────────────────────────────────
# [OPT-1] Frame-level projection cache
# ─────────────────────────────────────────────

@dataclass
class FrameCache:
    u: np.ndarray
    v: np.ndarray
    world_idx: np.ndarray
    z_pc: np.ndarray
    z_img: Optional[np.ndarray]
    depth_keep: Optional[np.ndarray]
    pixel_id: np.ndarray


def build_frame_cache(
    world_pts: np.ndarray,
    T_cw: np.ndarray,
    w: int, h: int,
    fx: float, fy: float, cx: float, cy: float,
    depth_raw: Optional[np.ndarray] = None,
    depth_scale: float = 1000.0,
    use_depth_consistency: bool = False,
    depth_tau_min: float = 0.05,
    depth_tau_k: float = 3.0,
    depth_tau_front: float = 0.10,
    voxel_size: float = 0.0,
) -> FrameCache:
    pts_cam = apply_T(T_cw, world_pts)
    z = pts_cam[:, 2]
    valid = (z > 0) & np.isfinite(z) & np.isfinite(pts_cam).all(axis=1)

    pts_v = pts_cam[valid]
    z_v = np.maximum(z[valid], 1e-6).astype(np.float32)
    idx_v = np.where(valid)[0].astype(np.int64)

    u_f = pts_v[:, 0] * fx / z_v + cx
    v_f = pts_v[:, 1] * fy / z_v + cy
    ok = np.isfinite(u_f) & np.isfinite(v_f)
    u_i = np.round(u_f[ok]).astype(np.int32)
    v_i = np.round(v_f[ok]).astype(np.int32)
    z_v = z_v[ok]
    idx_v = idx_v[ok]

    inside = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    u_i, v_i, z_v, idx_v = u_i[inside], v_i[inside], z_v[inside], idx_v[inside]

    pixel_id = v_i.astype(np.int64) * w + u_i.astype(np.int64)
    order = np.argsort(z_v, kind="stable")
    _, first = np.unique(pixel_id[order], return_index=True)
    sel = order[first]

    u_f2 = u_i[sel]
    v_f2 = v_i[sel]
    world_idx = idx_v[sel]
    z_pc = z_v[sel]
    pixel_id = pixel_id[order][first]

    if voxel_size > 0.0:
        keep = voxel_cap_indices_fast(np.arange(len(world_idx), dtype=np.int64),
                                      world_pts[world_idx], voxel_size)
        u_f2 = u_f2[keep]
        v_f2 = v_f2[keep]
        world_idx = world_idx[keep]
        z_pc = z_pc[keep]
        pixel_id = pixel_id[keep]

    z_img_arr: Optional[np.ndarray] = None
    depth_keep: Optional[np.ndarray] = None

    if use_depth_consistency and depth_raw is not None:
        z_img_arr = depth_raw[v_f2, u_f2].astype(np.float32) / float(depth_scale)
        valid_d = np.isfinite(z_img_arr) & (z_img_arr > 0)

        delta = z_pc - z_img_arr
        d0 = delta[valid_d]
        if d0.size > 0:
            med = float(np.median(d0))
            mad = float(np.median(np.abs(d0 - med)))
            tau = max(float(depth_tau_min), float(depth_tau_k) * 1.4826 * mad)
        else:
            med, tau = 0.0, float(depth_tau_min)

        keep_d = valid_d & (np.abs(delta - med) < tau)
        if depth_tau_front > 0:
            keep_d &= (z_pc < (z_img_arr + float(depth_tau_front)))
        depth_keep = keep_d

    return FrameCache(
        u=u_f2, v=v_f2, world_idx=world_idx, z_pc=z_pc,
        z_img=z_img_arr, depth_keep=depth_keep, pixel_id=pixel_id,
    )


def query_mask_from_cache(
    cache: FrameCache,
    mask01: np.ndarray,
    use_depth_consistency: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    in_mask = mask01[cache.v, cache.u] > 0

    if not use_depth_consistency:
        fg_idx = cache.world_idx[in_mask]
        return fg_idx, fg_idx

    valid_d = np.isfinite(cache.z_img) & (cache.z_img > 0)
    visible_mask = in_mask & valid_d
    if not np.any(visible_mask):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    mask_visible_idx = cache.world_idx[visible_mask]
    fg_idx = cache.world_idx[in_mask & cache.depth_keep]
    return mask_visible_idx, fg_idx


# ─────────────────────────────────────────────
# [OPT-2] SemanticGroup
# ─────────────────────────────────────────────

@dataclass
class SemanticGroup:
    typ: str
    name: str

    raw_names: set = field(default_factory=set)
    observations: int = field(default=0)
    sample_masks: List[str] = field(default_factory=list)

    _vis_keys: List[np.ndarray] = field(default_factory=list)
    _vis_cnts: List[np.ndarray] = field(default_factory=list)
    _fg_keys:  List[np.ndarray] = field(default_factory=list)
    _fg_cnts:  List[np.ndarray] = field(default_factory=list)

    _vis_count: Optional[dict] = field(default=None)
    _fg_count:  Optional[dict] = field(default=None)
    _indices:   Optional[set]  = field(default=None)

    def add_obs(
        self,
        raw_name: str,
        mask_name: str,
        visible_indices: np.ndarray,
        fg_indices: np.ndarray,
        max_sample_masks: int = 10,
    ):
        self.raw_names.add(normalize_name(raw_name))
        self.observations += 1
        if mask_name and len(self.sample_masks) < max_sample_masks:
            self.sample_masks.append(mask_name)

        if visible_indices is not None and len(visible_indices) > 0:
            vis = np.asarray(visible_indices, dtype=np.int64).ravel()
            uq, cnt = np.unique(vis, return_counts=True)
            self._vis_keys.append(uq)
            self._vis_cnts.append(cnt.astype(np.int32))
            self._indices = None

        if fg_indices is not None and len(fg_indices) > 0:
            fg = np.asarray(fg_indices, dtype=np.int64).ravel()
            uq, cnt = np.unique(fg, return_counts=True)
            self._fg_keys.append(uq)
            self._fg_cnts.append(cnt.astype(np.int32))

    def _build_counts(self):
        if self._vis_count is not None:
            return

        def merge(keys_list, cnts_list):
            if not keys_list:
                return {}
            all_k = np.concatenate(keys_list)
            all_c = np.concatenate(cnts_list)
            order = np.argsort(all_k, kind="stable")
            all_k = all_k[order]
            all_c = all_c[order]
            uq, _, inv = np.unique(all_k, return_index=True, return_inverse=True)
            totals = np.zeros(len(uq), dtype=np.int64)
            np.add.at(totals, inv, all_c)
            return dict(zip(uq.tolist(), totals.tolist()))

        self._vis_count = merge(self._vis_keys, self._vis_cnts)
        self._fg_count  = merge(self._fg_keys,  self._fg_cnts)
        self._indices   = set(self._vis_count.keys())

    @property
    def vis_count(self) -> dict:
        self._build_counts()
        return self._vis_count  # type: ignore

    @property
    def fg_count(self) -> dict:
        self._build_counts()
        return self._fg_count   # type: ignore

    @property
    def indices(self) -> set:
        self._build_counts()
        return self._indices    # type: ignore

    def all_names(self) -> List[str]:
        return sorted({normalize_name(x) for x in (self.raw_names | {self.name}) if x})


# ─────────────────────────────────────────────
# [OPT-3] Vectorized vote filter
# ─────────────────────────────────────────────

def filter_kept_vectorized(
    g: SemanticGroup,
    min_vis_eff: int,
    min_fg: int,
    use_wilson: bool,
    wilson_p0: float,
    wilson_z: float,
    min_fg_ratio: float,
) -> set:
    if not g.indices:
        return set()

    idx_arr = np.fromiter(g.indices, dtype=np.int64)
    vis_d = g.vis_count
    fg_d = g.fg_count

    vcnt = np.array([vis_d.get(int(i), 0) for i in idx_arr], dtype=np.int64)
    fcnt = np.array([fg_d.get(int(i), 0) for i in idx_arr], dtype=np.int64)

    mask = vcnt >= min_vis_eff
    if min_fg > 0:
        mask &= (fcnt >= min_fg)

    vcnt_m = vcnt[mask]
    fcnt_m = fcnt[mask]
    idx_m = idx_arr[mask]
    if idx_m.size == 0:
        return set()

    if use_wilson:
        lb = wilson_lower_bound_vec(fcnt_m, vcnt_m, wilson_z)
        sub = lb >= wilson_p0
    else:
        ratio = np.where(vcnt_m > 0, fcnt_m / vcnt_m.astype(np.float64), 0.0)
        sub = ratio >= min_fg_ratio

    return set(idx_m[sub].tolist())


# ─────────────────────────────────────────────
# Scene/sequence discovery
# ─────────────────────────────────────────────

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


def find_transform_npy(seq_dir: Path) -> Optional[Path]:
    cands = sorted(list(seq_dir.glob("*_transform.npy")), key=lambda p: natural_key(p.name))
    return cands[0] if cands else None


def find_traj_path(seq_dir: Path) -> Path:
    p = seq_dir / "hires_poses.traj"
    if not p.exists():
        raise FileNotFoundError(f"Trajectory not found: {p}")
    return p


def find_intr_dir(seq_dir: Path) -> Path:
    p = seq_dir / "hires_wide_intrinsics"
    if not p.exists():
        raise FileNotFoundError(f"Intrinsics dir not found: {p}")
    return p


# ─────────────────────────────────────────────
# Point cloud loading
# ─────────────────────────────────────────────

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


def auto_find_pcd(scene_root: Path, seq_dir: Path) -> Optional[Path]:
    def is_bad_npy(p: Path) -> bool:
        n = p.name.lower()
        return any(x in n for x in ["_transform", "pose", "traj", "intrinsic"])

    def collect(d: Path) -> List[Path]:
        if not d.exists():
            return []
        cands = [p for pat in ["*.ply", "*.pcd", "*.npz", "*.npy"]
                 for p in d.glob(pat) if p.is_file()]
        return [p for p in cands if not (p.suffix.lower() == ".npy" and is_bad_npy(p))]

    def score(p: Path):
        n = p.name.lower()
        s = {".ply": 5, ".pcd": 5, ".npz": 4, ".npy": 1}.get(p.suffix.lower(), 0)
        for kw, v in [("laser", 5), ("scan", 4), ("point", 3), ("pcd", 3), ("cloud", 2)]:
            if kw in n:
                s += v
        if "color" in n or "rgb" in n:
            s -= 1
        return (-s, natural_key(p.name))

    cands = collect(scene_root) or collect(seq_dir)
    return sorted(cands, key=score)[0] if cands else None


# ─────────────────────────────────────────────
# DBSCAN helpers
# ─────────────────────────────────────────────

def dbscan_largest_cluster_centroid(pts, eps, min_samples):
    if pts is None or pts.size == 0:
        return None, {"num_points": 0, "num_clusters": 0, "largest_cluster_points": 0,
                      "largest_cluster_ratio": 0.0, "noise_points": 0}
    pts = np.asarray(pts, dtype=np.float64)
    n = int(pts.shape[0])
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_samples),
                                              print_progress=False), dtype=np.int32)
        num_noise = int(np.sum(labels < 0))
        valid = labels >= 0
        if not np.any(valid):
            return pts.mean(axis=0).astype(np.float32), \
                   {"num_points": n, "num_clusters": 0, "largest_cluster_points": 0,
                    "largest_cluster_ratio": 0.0, "noise_points": num_noise}
        ids, counts = np.unique(labels[valid], return_counts=True)
        best_id = int(ids[np.argmax(counts)])
        best_cnt = int(counts.max())
        c = pts[labels == best_id].mean(axis=0)
        info = {"num_points": n, "num_clusters": int(len(ids)),
                "largest_cluster_points": best_cnt,
                "largest_cluster_ratio": float(best_cnt) / n,
                "noise_points": num_noise}
        return c.astype(np.float32), info
    except Exception:
        return pts.mean(axis=0).astype(np.float32), \
               {"num_points": n, "num_clusters": 0, "largest_cluster_points": n,
                "largest_cluster_ratio": 1.0, "noise_points": 0}


def split_into_instances_by_dbscan(world_pts, idxset, eps, min_samples):
    if not idxset:
        return []
    idx_list = np.array(sorted(map(int, idxset)), dtype=np.int64)
    pts = world_pts[idx_list]
    if pts.shape[0] == 0:
        return []
    try:
        import open3d as o3d
    except Exception:
        return [set(idx_list.tolist())]
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_samples),
                                              print_progress=False), dtype=np.int32)
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return [set(idx_list.tolist())]
        cluster_ids, counts = np.unique(labels[valid_mask], return_counts=True)
        result = []
        for cid in cluster_ids[np.argsort(-counts)]:
            ci = idx_list[labels == cid]
            if ci.size > 0:
                result.append(set(ci.tolist()))
        return result if result else [set(idx_list.tolist())]
    except Exception:
        return [set(idx_list.tolist())]


# ─────────────────────────────────────────────
# AABB helpers (for object-only bbox merge)
# ─────────────────────────────────────────────

def compute_aabb_from_indices(world_pts: np.ndarray, idxset: set):
    if not idxset:
        return None
    idx_arr = np.array(sorted(map(int, idxset)), dtype=np.int64)
    if idx_arr.size == 0:
        return None
    pts = world_pts[idx_arr]
    if pts.shape[0] == 0:
        return None
    bmin = pts.min(axis=0).astype(np.float64)
    bmax = pts.max(axis=0).astype(np.float64)
    return bmin, bmax


def compute_aabb_from_points(pts: np.ndarray):
    if pts is None or pts.size == 0:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    return bmin, bmax


def aabb_volume(bmin: np.ndarray, bmax: np.ndarray) -> float:
    ext = np.maximum(bmax - bmin, 0.0)
    return float(ext[0] * ext[1] * ext[2])


def aabb_intersection(bmin1: np.ndarray, bmax1: np.ndarray,
                      bmin2: np.ndarray, bmax2: np.ndarray):
    inter_min = np.maximum(bmin1, bmin2)
    inter_max = np.minimum(bmax1, bmax2)
    ext = np.maximum(inter_max - inter_min, 0.0)
    return inter_min, inter_max, float(ext[0] * ext[1] * ext[2])


def aabb_iou_and_containment(bmin1: np.ndarray, bmax1: np.ndarray,
                             bmin2: np.ndarray, bmax2: np.ndarray):
    v1 = aabb_volume(bmin1, bmax1)
    v2 = aabb_volume(bmin2, bmax2)
    _, _, inter = aabb_intersection(bmin1, bmax1, bmin2, bmax2)
    union = max(v1 + v2 - inter, 1e-12)
    iou = inter / union
    contain_small = inter / max(min(v1, v2), 1e-12)
    contain_1in2 = inter / max(v1, 1e-12)
    contain_2in1 = inter / max(v2, 1e-12)
    return {
        "vol1": float(v1),
        "vol2": float(v2),
        "inter": float(inter),
        "iou": float(iou),
        "contain_small": float(contain_small),
        "contain_1in2": float(contain_1in2),
        "contain_2in1": float(contain_2in1),
    }


# ─────────────────────────────────────────────
# Affordance assignment
# ─────────────────────────────────────────────

def create_build_scene_graph_config(args, scene_id, seq_id, pcd_path, transform_path, name_map_path, depth_dir):
    """Create comprehensive experiment configuration for build_scene_graph.py"""
    config = {
        'experiment': {
            'timestamp': datetime.now().isoformat(),
            'script_name': 'build_scene_graph.py',
            'version': '2.0.0',
            'description': 'Build 3D scene graph with optimized frame-cache, vectorized vote, object-bbox-merge, and affordance-priority ownership'
        },
        'scene_data': {
            'scene_id': scene_id,
            'sequence_id': seq_id,
            'pcd_path': str(pcd_path),
            'transform_npy': str(transform_path) if transform_path else None,
            'name_map_json': str(name_map_path) if name_map_path else None,
            'depth_dir': str(depth_dir) if depth_dir else None,
        },
        'input_paths': {
            'masks_root': args.masks_root,
            'data_root': args.data_root,
            'out_root': args.out_root if args.out_root.strip() else None,
            'out_json': args.out_json if args.out_json.strip() else None,
            'ply_dir': args.ply_dir if args.ply_dir.strip() else 'auto',
        },
        'point_cloud': {
            'pcd_apply_transform': args.pcd_apply_transform,
            'pcd_transform_invert': args.pcd_transform_invert,
            'voxel_size': args.voxel_size,
        },
        'camera_parameters': {
            'frame_stride': args.frame_stride,
            'transform_invert': args.transform_invert,
            'disable_transform': args.disable_transform,
        },
        'depth_consistency': {
            'use_depth_consistency': args.use_depth_consistency,
            'depth_scale': args.depth_scale,
            'depth_tau_min': args.depth_tau_min,
            'depth_tau_k': args.depth_tau_k,
            'depth_tau_front': args.depth_tau_front,
        },
        'clustering': {
            'use_dbscan_position': args.use_dbscan_position,
            'dbscan_eps': args.dbscan_eps,
            'dbscan_min_samples': args.dbscan_min_samples,
            'obj_dbscan_eps': args.obj_dbscan_eps if args.obj_dbscan_eps > 0 else args.dbscan_eps,
            'aff_dbscan_eps': args.aff_dbscan_eps if args.aff_dbscan_eps > 0 else args.dbscan_eps,
        },
        'filtering': {
            'use_wilson': args.use_wilson,
            'wilson_p0': args.wilson_p0,
            'wilson_z': args.wilson_z,
            'min_vis': args.min_vis,
            'min_vis_frac': args.min_vis_frac,
            'min_fg': args.min_fg,
            'min_fg_ratio': args.min_fg_ratio,
        },
        'affordance_filtering': {
            'aff_min_vis': args.aff_min_vis,
            'aff_min_vis_frac': args.aff_min_vis_frac,
            'aff_min_fg': args.aff_min_fg,
            'aff_min_fg_ratio': args.aff_min_fg_ratio,
            'aff_use_wilson': args.aff_use_wilson,
            'aff_wilson_p0': args.aff_wilson_p0,
            'aff_wilson_z': args.aff_wilson_z,
        },
        'node_merging': {
            'merge_nodes': args.merge_nodes,
            'merge_min_common': args.merge_min_common,
            'merge_min_ratio': args.merge_min_ratio,
            'merge_object_bbox': args.merge_object_bbox,
            'bbox_iou_thr': args.bbox_iou_thr,
            'bbox_contain_thr': args.bbox_contain_thr,
        },
        'ownership': {
            'enforce_unique_ownership': args.enforce_unique_ownership,
            'recompute_position_after_prune': args.recompute_position_after_prune,
            'rule': 'affordance_priority',
        },
        'output': {
            'save_ply': args.save_ply,
            'ply_apply_T_out': args.ply_apply_T_out,
            'ply_voxel_down': args.ply_voxel_down,
            'debug': args.debug,
            'debug_frames': args.debug_frames,
            'log_every': args.log_every,
        }
    }
    return config


def save_build_scene_graph_config(config, output_dir):
    """Save experiment configuration to YAML file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = config['experiment']['timestamp'].replace(':', '-')
    config_file = os.path.join(output_dir, f"build_scene_graph_config_{timestamp_str}.yaml")
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[CONFIG] Build scene graph configuration saved to: {config_file}")
    return config_file


def assign_affordances_to_nearest_object(nodes: List[dict]) -> List[dict]:
    objects = [n for n in nodes if n.get("type") == "object"]
    affords = [n for n in nodes if n.get("type") == "affordance"]
    if not affords:
        return nodes
    if not objects:
        for a in affords:
            a["belong_to"] = None
        return nodes
    obj_pos = np.array([o["position"] for o in objects], dtype=np.float64)
    obj_ids = [o["id"] for o in objects]
    for a in affords:
        ap = np.array(a["position"], dtype=np.float64)
        a["belong_to"] = int(obj_ids[int(np.argmin(np.linalg.norm(obj_pos - ap, axis=1)))])
    return nodes


# ─────────────────────────────────────────────
# [OPT-5] Merge by point-index overlap
# ─────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def merge_nodes_by_index_overlap(
    nodes_with_indices, world_pts, T_out,
    use_dbscan_position, obj_dbscan_eps, aff_dbscan_eps, dbscan_min_samples,
    min_common=50, min_ratio=0.3,
):
    if not nodes_with_indices:
        return nodes_with_indices, {"enabled": True, "groups_considered": 0,
                                    "nodes_before": 0, "nodes_after": 0,
                                    "merged_components": 0,
                                    "min_common": int(min_common), "min_ratio": float(min_ratio)}

    buckets: Dict[Tuple[str, str], List] = {}
    for node, idxset in nodes_with_indices:
        buckets.setdefault((node.get("type", ""), node.get("name", "")), []).append((node, idxset))

    merged_all = []
    groups_considered = merged_components = 0

    for key, lst in buckets.items():
        if len(lst) == 1:
            merged_all.append(lst[0])
            continue

        groups_considered += 1
        n = len(lst)
        sizes = [len(s) for _, s in lst]

        all_pts_list, all_nid_list = [], []
        for i, (_, idxset) in enumerate(lst):
            arr = np.fromiter(idxset, dtype=np.int64)
            all_pts_list.append(arr)
            all_nid_list.append(np.full(len(arr), i, dtype=np.int32))

        all_pts = np.concatenate(all_pts_list)
        all_nid = np.concatenate(all_nid_list)
        order = np.argsort(all_pts, kind="stable")
        all_pts = all_pts[order]
        all_nid = all_nid[order]

        boundaries = np.where(np.diff(all_pts) != 0)[0] + 1
        groups_of_nodes = np.split(all_nid, boundaries)

        pair_count: Dict[Tuple[int, int], int] = {}
        for g_nodes in groups_of_nodes:
            if len(g_nodes) < 2:
                continue
            for a, b in combinations(sorted(set(g_nodes.tolist())), 2):
                pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

        uf = UnionFind(n)
        for (ia, ib), inter in pair_count.items():
            if inter < int(min_common):
                continue
            denom = min(sizes[ia], sizes[ib])
            if denom > 0 and float(inter) / denom >= float(min_ratio):
                uf.union(ia, ib)

        comp: Dict[int, List[int]] = {}
        for i in range(n):
            comp.setdefault(uf.find(i), []).append(i)

        if len(comp) == n:
            merged_all.extend(lst)
            continue

        merged_components += n - len(comp)

        for root, members in comp.items():
            if len(members) == 1:
                merged_all.append(lst[members[0]])
                continue

            merged_set: set = set()
            obs_sum = 0
            sample_masks = []
            names_union = set()
            for mi in members:
                node_i, idx_i = lst[mi]
                merged_set |= set(idx_i)
                obs_sum += int(node_i.get("confidence", {}).get("observations", 0))
                sample_masks.extend(node_i.get("debug", {}).get("sample_masks", [])[:3])
                for x in node_i.get("names", []):
                    names_union.add(str(x))

            idx_arr = np.array(sorted(map(int, merged_set)), dtype=np.int64)
            pts_sel = world_pts[idx_arr].astype(np.float32) if idx_arr.size > 0 else np.zeros((0, 3), np.float32)
            if pts_sel.shape[0] == 0:
                continue

            base_node = dict(lst[members[0]][0])
            typ0 = base_node.get("type", "")
            eps_here = float(aff_dbscan_eps) if typ0 == "affordance" else float(obj_dbscan_eps)

            if use_dbscan_position:
                c_db, dbinfo = dbscan_largest_cluster_centroid(pts_sel, eps_here, dbscan_min_samples)
                c_world = c_db if c_db is not None else pts_sel.mean(axis=0)
            else:
                c_world = pts_sel.mean(axis=0)
                dbinfo = None

            c_out = apply_T(T_out, np.asarray(c_world).reshape(1, 3)).reshape(3)
            aabb = compute_aabb_from_points(pts_sel)

            base_node["position"] = c_out.tolist()
            base_node["names"] = sorted({normalize_name(x) for x in names_union if x})
            base_node["confidence"] = {"points_final": int(len(merged_set)), "observations": int(obs_sum)}
            base_node.setdefault("debug", {})
            base_node["debug"].update({
                "merged_from_local_ids": members,
                "merge_rule": {"min_common": int(min_common), "min_ratio": float(min_ratio)},
                "sample_masks": sample_masks[:10],
                "dbscan_position": dbinfo,
                "bbox_world": {
                    "min": aabb[0].tolist(),
                    "max": aabb[1].tolist(),
                } if aabb is not None else None,
            })
            merged_all.append((base_node, merged_set))

    info = {"enabled": True, "groups_considered": groups_considered,
            "nodes_before": len(nodes_with_indices), "nodes_after": len(merged_all),
            "merged_components": merged_components,
            "min_common": int(min_common), "min_ratio": float(min_ratio)}
    return merged_all, info


# ─────────────────────────────────────────────
# [MOD-1] Object-only bbox merge
# ─────────────────────────────────────────────

def merge_object_nodes_by_bbox(
    nodes_with_indices,
    world_pts,
    T_out,
    use_dbscan_position,
    obj_dbscan_eps,
    dbscan_min_samples,
    bbox_iou_thr=0.25,
    bbox_contain_thr=0.80,
):
    """
    Merge ONLY object nodes if their 3D AABBs have:
      - sufficiently large IoU, or
      - strong containment relation.

    Important:
      - Only object nodes are considered.
      - Only same-name objects are allowed to merge.
    """
    if not nodes_with_indices:
        return nodes_with_indices, {
            "enabled": True,
            "mode": "object_only_bbox_merge",
            "nodes_before": 0,
            "nodes_after": 0,
            "merged_components": 0,
            "bbox_iou_thr": float(bbox_iou_thr),
            "bbox_contain_thr": float(bbox_contain_thr),
        }

    object_items = []
    non_object_items = []

    for node, idxset in nodes_with_indices:
        if node.get("type") == "object":
            object_items.append((node, idxset))
        else:
            non_object_items.append((node, idxset))

    n = len(object_items)
    if n <= 1:
        return nodes_with_indices, {
            "enabled": True,
            "mode": "object_only_bbox_merge",
            "nodes_before": len(nodes_with_indices),
            "nodes_after": len(nodes_with_indices),
            "merged_components": 0,
            "bbox_iou_thr": float(bbox_iou_thr),
            "bbox_contain_thr": float(bbox_contain_thr),
            "same_name_only": True,
        }

    aabbs = []
    valid_mask = []
    for node, idxset in object_items:
        box = compute_aabb_from_indices(world_pts, idxset)
        aabbs.append(box)
        valid_mask.append(box is not None)

    uf = UnionFind(n)

    for i in range(n):
        if not valid_mask[i]:
            continue
        bmin_i, bmax_i = aabbs[i]
        name_i = object_items[i][0].get("name", "")

        for j in range(i + 1, n):
            if not valid_mask[j]:
                continue

            name_j = object_items[j][0].get("name", "")

            if name_i != name_j:
                continue

            bmin_j, bmax_j = aabbs[j]
            rel = aabb_iou_and_containment(bmin_i, bmax_i, bmin_j, bmax_j)

            should_merge = (
                rel["iou"] >= float(bbox_iou_thr)
                or rel["contain_small"] >= float(bbox_contain_thr)
            )

            if should_merge:
                uf.union(i, j)

    comp = {}
    for i in range(n):
        comp.setdefault(uf.find(i), []).append(i)

    merged_objects = []
    merged_components = 0

    for root, members in comp.items():
        if len(members) == 1:
            merged_objects.append(object_items[members[0]])
            continue

        merged_components += len(members) - 1

        merged_set = set()
        obs_sum = 0
        sample_masks = []
        names_union = set()

        for mi in members:
            node_i, idx_i = object_items[mi]
            merged_set |= set(idx_i)
            obs_sum += int(node_i.get("confidence", {}).get("observations", 0))
            sample_masks.extend(node_i.get("debug", {}).get("sample_masks", [])[:3])
            for x in node_i.get("names", []):
                names_union.add(str(x))

        idx_arr = np.array(sorted(map(int, merged_set)), dtype=np.int64)
        pts_sel = world_pts[idx_arr].astype(np.float32)
        if pts_sel.shape[0] == 0:
            continue

        if use_dbscan_position:
            c_db, dbinfo = dbscan_largest_cluster_centroid(
                pts_sel, float(obj_dbscan_eps), int(dbscan_min_samples)
            )
            c_world = c_db if c_db is not None else pts_sel.mean(axis=0)
        else:
            c_world = pts_sel.mean(axis=0)
            dbinfo = None

        c_out = apply_T(T_out, np.asarray(c_world).reshape(1, 3)).reshape(3)
        aabb = compute_aabb_from_points(pts_sel)

        base_node = dict(object_items[members[0]][0])
        base_node["position"] = c_out.tolist()
        base_node["names"] = sorted({normalize_name(x) for x in names_union if x})
        base_node["confidence"] = {
            "points_final": int(len(merged_set)),
            "observations": int(obs_sum),
        }
        base_node.setdefault("debug", {})
        base_node["debug"].update({
            "bbox_merged_from_local_ids": members,
            "bbox_merge_rule": {
                "bbox_iou_thr": float(bbox_iou_thr),
                "bbox_contain_thr": float(bbox_contain_thr),
                "same_name_only": True,
            },
            "sample_masks": sample_masks[:10],
            "dbscan_position": dbinfo,
            "bbox_world": {
                "min": aabb[0].tolist(),
                "max": aabb[1].tolist(),
            } if aabb is not None else None,
        })

        merged_objects.append((base_node, merged_set))

    merged_all = merged_objects + non_object_items

    info = {
        "enabled": True,
        "mode": "object_only_bbox_merge",
        "nodes_before": len(nodes_with_indices),
        "nodes_after": len(merged_all),
        "merged_components": int(merged_components),
        "bbox_iou_thr": float(bbox_iou_thr),
        "bbox_contain_thr": float(bbox_contain_thr),
        "same_name_only": True,
    }
    return merged_all, info


# ─────────────────────────────────────────────
# [MOD-2] Ownership: affordance priority
# ─────────────────────────────────────────────

def enforce_affordance_priority_ownership(
    nodes_with_indices, world_pts, T_out,
    use_dbscan_position, obj_dbscan_eps, dbscan_min_samples,
):
    """
    Affordance priority:
      if point in (object ∩ affordance), keep only in affordance.
    """
    if not nodes_with_indices:
        return nodes_with_indices, {
            "enabled": True,
            "objects_before": 0,
            "affordances": 0,
            "removed_points": 0,
            "dropped_object_nodes": 0,
            "rule": "affordance_priority",
        }

    aff_all = set()
    obj_nodes = 0
    aff_nodes = 0

    for node, idxset in nodes_with_indices:
        if node.get("type") == "affordance":
            aff_nodes += 1
            aff_all |= set(map(int, idxset))

    removed_points = 0
    dropped = 0
    kept = []

    for node, idxset in nodes_with_indices:
        if node.get("type") != "object":
            kept.append((node, idxset))
            continue

        obj_nodes += 1
        before = len(idxset)
        pruned = set(map(int, idxset)) - aff_all
        removed_points += before - len(pruned)

        if not pruned:
            dropped += 1
            continue

        idx_arr = np.array(sorted(pruned), dtype=np.int64)
        pts_sel = world_pts[idx_arr].astype(np.float32)
        if pts_sel.shape[0] == 0:
            dropped += 1
            continue

        if use_dbscan_position:
            c_db, info = dbscan_largest_cluster_centroid(
                pts_sel, obj_dbscan_eps, dbscan_min_samples
            )
            c_world = c_db if c_db is not None else pts_sel.mean(axis=0)
            node.setdefault("debug", {})["dbscan_position_after_prune"] = info
        else:
            c_world = pts_sel.mean(axis=0)

        c_out = apply_T(T_out, np.asarray(c_world).reshape(1, 3)).reshape(3)
        aabb = compute_aabb_from_points(pts_sel)

        node["position"] = c_out.tolist()
        node.setdefault("confidence", {})["points_final"] = int(len(pruned))
        node.setdefault("debug", {})["ownership_prune"] = {
            "rule": "affordance_priority",
            "removed_points": int(before - len(pruned)),
            "points_before": int(before),
            "points_after": int(len(pruned)),
        }
        node["debug"]["bbox_world_after_prune"] = {
            "min": aabb[0].tolist(),
            "max": aabb[1].tolist(),
        } if aabb is not None else None

        kept.append((node, pruned))

    info = {
        "enabled": True,
        "objects_before": int(obj_nodes),
        "affordances": int(aff_nodes),
        "removed_points": int(removed_points),
        "dropped_object_nodes": int(dropped),
        "objects_after": int(obj_nodes - dropped),
        "rule": "if point in (object ∩ affordance) => keep only in affordance",
    }
    return kept, info


# ─────────────────────────────────────────────
# PLY saving + node colors
# ─────────────────────────────────────────────

def safe_filename(s: str) -> str:
    s = normalize_name(s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _fnv1a_32(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def color_for_node_key(typ, name, inst_id):
    h32 = _fnv1a_32(f"{typ}|{name}|{inst_id}".encode())
    return _hsv_to_rgb((h32 & 0xFFFF) / 65536.0, 0.75, 0.95)


def save_points_as_ply(points_xyz, out_path, color_rgb01=None, voxel_down=0.0):
    if points_xyz is None or points_xyz.size == 0:
        return
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("Saving PLY requires open3d.") from e
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if color_rgb01 is not None:
        c = np.tile(np.array(color_rgb01).reshape(1, 3), (len(points_xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(c)
    if voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(voxel_down))
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)


def save_xyz_as_ply(points_xyz, out_path, colors_rgb01=None):
    if points_xyz is None or points_xyz.size == 0:
        return
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("Saving PLY requires open3d.") from e
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    if colors_rgb01 is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb01, dtype=np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build 3D scene graph — OPTIMIZED (frame-cache, vectorized vote, numpy merge, object-bbox-merge, affordance-priority ownership)."
    )
    ap.add_argument("--scene_id", type=int, default=-1)
    ap.add_argument("--masks_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--pcd_path", type=str, default="")
    ap.add_argument("--pcd_apply_transform", type=str, default="")
    ap.add_argument("--pcd_transform_invert", action="store_true")
    ap.add_argument("--name_map_json", type=str, default="")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--transform_invert", action="store_true")
    ap.add_argument("--disable_transform", action="store_true")
    ap.add_argument("--out_json", type=str, default="", help="Output JSON file path (deprecated, use --out_root instead)")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory (will create scene_id subdirectory)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_frames", type=int, default=3)
    ap.add_argument("--use_depth_consistency", action="store_true")
    ap.add_argument("--depth_dir", type=str, default="")
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_tau_min", type=float, default=0.05)
    ap.add_argument("--depth_tau_k", type=float, default=3.0)
    ap.add_argument("--depth_tau_front", type=float, default=0.10)
    ap.add_argument("--use_dbscan_position", action="store_true")
    ap.add_argument("--dbscan_eps", type=float, default=0.10)
    ap.add_argument("--dbscan_min_samples", type=int, default=10)
    ap.add_argument("--save_ply", action="store_true")
    ap.add_argument("--ply_dir", type=str, default="")
    ap.add_argument("--ply_apply_T_out", action="store_true")
    ap.add_argument("--ply_voxel_down", type=float, default=0.0)
    ap.add_argument("--use_wilson", action="store_true")
    ap.add_argument("--wilson_p0", type=float, default=0.70)
    ap.add_argument("--wilson_z", type=float, default=1.96)
    ap.add_argument("--min_vis", type=int, default=3)
    ap.add_argument("--min_vis_frac", type=float, default=0.0)
    ap.add_argument("--min_fg", type=int, default=0)
    ap.add_argument("--min_fg_ratio", type=float, default=0.70)
    ap.add_argument("--voxel_size", type=float, default=0.02)
    ap.add_argument("--obj_dbscan_eps", type=float, default=-1.0)
    ap.add_argument("--aff_dbscan_eps", type=float, default=-1.0)

    # legacy flags kept for compatibility, but ownership is now affordance-priority
    ap.add_argument("--enforce_unique_ownership", action="store_true")
    ap.add_argument("--recompute_position_after_prune", action="store_true")

    ap.add_argument("--merge_nodes", action="store_true")
    ap.add_argument("--merge_min_common", type=int, default=50)
    ap.add_argument("--merge_min_ratio", type=float, default=0.30)

    # [MOD-1] object-only bbox merge
    ap.add_argument("--merge_object_bbox", action="store_true")
    ap.add_argument("--bbox_iou_thr", type=float, default=0.25)
    ap.add_argument("--bbox_contain_thr", type=float, default=0.80)

    ap.add_argument("--aff_min_vis", type=int, default=-1)
    ap.add_argument("--aff_min_vis_frac", type=float, default=-1.0)
    ap.add_argument("--aff_min_fg", type=int, default=-1)
    ap.add_argument("--aff_min_fg_ratio", type=float, default=-1.0)
    ap.add_argument("--aff_use_wilson", action="store_true")
    ap.add_argument("--aff_wilson_p0", type=float, default=0.50)
    ap.add_argument("--aff_wilson_z", type=float, default=1.96)
    args = ap.parse_args()

    obj_dbscan_eps = float(args.obj_dbscan_eps) if args.obj_dbscan_eps > 0 else float(args.dbscan_eps)
    aff_dbscan_eps = float(args.aff_dbscan_eps) if args.aff_dbscan_eps > 0 else float(args.dbscan_eps)
    print(f"[DBSCAN] eps: object={obj_dbscan_eps}  affordance={aff_dbscan_eps}")

    voxel_size = float(args.voxel_size)

    name_map_path = Path(args.name_map_json) if args.name_map_json.strip() else None
    name_map = load_name_map_json(name_map_path) if name_map_path else {}
    if name_map_path:
        print(f"[NameMap] loaded {len(name_map)} mappings from: {name_map_path}")

    masks_root_in = Path(args.masks_root)
    data_root = Path(args.data_root)
    
    # Handle output path logic
    if args.out_root.strip():
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        out_json = None  # Will be set per scene
    elif args.out_json.strip():
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_root = None
    else:
        raise ValueError("Either --out_root or --out_json must be specified")

    if args.scene_id != -1:
        sid_list = [str(args.scene_id)]
        masks_roots = {str(args.scene_id): masks_root_in}
    else:
        if not masks_root_in.exists():
            raise FileNotFoundError(f"Masks root not found: {masks_root_in}")
        sid_list = []
        masks_roots = {}
        for p in masks_root_in.iterdir():
            if p.is_dir() and re.fullmatch(r"\d+", p.name):
                m = p / "masks"
                if m.exists():
                    sid_list.append(p.name)
                    masks_roots[p.name] = m
        sid_list = sorted(sid_list, key=natural_key)
        if not sid_list:
            raise ValueError("No scenes found.")

    for sid in sid_list:
        print(f"\n{'='*60}\nProcessing scene: {sid}\n{'='*60}")

        scene_id = int(sid)
        masks_root = masks_roots[sid]
        if out_root is not None:
            # New behavior: use out_root with scene subdirectories
            scene_out_dir = out_root / str(scene_id)
            scene_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = scene_out_dir / "scene_graph.json"
        else:
            # Legacy behavior: use out_json
            out_path = out_json if args.scene_id != -1 else (out_json.parent / f"{scene_id}_scene_graph.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)

        ply_dir = Path(args.ply_dir) if args.ply_dir.strip() else out_path.parent / "ply"
        if args.save_ply:
            if ply_dir.exists():
                for child in ply_dir.iterdir():
                    try:
                        child.unlink() if child.is_file() or child.is_symlink() else shutil.rmtree(child)
                    except Exception as e:
                        print(f"[PLY] Warning: failed to remove {child}: {e}")
            ply_dir.mkdir(parents=True, exist_ok=True)

        scene_root = data_root / sid
        seq_dir = find_first_sequence_dir(scene_root)
        seq_id = seq_dir.name
        print(f"Sequence: {seq_id}")

        intr_dir = find_intr_dir(seq_dir)
        traj_path = find_traj_path(seq_dir)
        intr_files = sorted([p for p in intr_dir.iterdir() if p.is_file()],
                            key=lambda p: natural_key(p.name))
        if not intr_files:
            raise ValueError(f"No intrinsic files in: {intr_dir}")

        print(f"Loading trajectory: {traj_path}")
        T_cw_list = read_traj_Twc(traj_path)
        print(f"Loaded {len(T_cw_list)} poses")

        T_out = np.eye(4, dtype=np.float64)
        transform_path = None
        if not args.disable_transform:
            transform_path = find_transform_npy(seq_dir)
            if transform_path and transform_path.exists():
                print(f"Loading transform: {transform_path}")
                T = np.asarray(np.load(str(transform_path)), dtype=np.float64)
                validate_transform(T, "Output transform")
                if args.transform_invert:
                    T = np.linalg.inv(T)
                    print("  (inverted)")
                T_out = T

        depth_files: Optional[List[Path]] = None
        depth_dir: Optional[Path] = None
        if args.use_depth_consistency:
            depth_dir = Path(args.depth_dir) if args.depth_dir.strip() else find_depth_dir_auto(seq_dir)
            if depth_dir is None or not depth_dir.exists():
                raise FileNotFoundError("use_depth_consistency=True but depth_dir not found.")
            depth_files = list_depth_files(depth_dir)
            if not depth_files:
                raise FileNotFoundError(f"No depth files under: {depth_dir}")
            print(f"[Depth] dir={depth_dir}  files={len(depth_files)}  scale={args.depth_scale}")

        frame_ids_all = list_frame_dirs(masks_root)
        frame_ids = frame_ids_all[::max(1, args.frame_stride)]
        max_idx = max(frame_ids)

        print(f"Frames: {len(frame_ids_all)} total, {len(frame_ids)} to process")

        if max_idx >= len(intr_files):
            raise ValueError(f"Frame index {max_idx} >= intrinsics count {len(intr_files)}")
        if max_idx >= len(T_cw_list):
            raise ValueError(f"Frame index {max_idx} >= trajectory count {len(T_cw_list)}")
        if args.use_depth_consistency and depth_files and max_idx >= len(depth_files):
            raise ValueError(f"Frame index {max_idx} >= depth files count {len(depth_files)}")

        pcd_path = Path(args.pcd_path) if args.pcd_path.strip() else auto_find_pcd(scene_root, seq_dir)
        if pcd_path is None:
            raise FileNotFoundError("No point cloud found. Specify --pcd_path")

        print(f"Loading point cloud: {pcd_path}")
        world_pts = load_pcd_points(pcd_path)
        print(f"Loaded {len(world_pts):,} points")

        if args.pcd_apply_transform.strip():
            T_pcd = np.asarray(np.load(args.pcd_apply_transform), dtype=np.float64)
            validate_transform(T_pcd, "PCD transform")
            if args.pcd_transform_invert:
                T_pcd = np.linalg.inv(T_pcd)
            world_pts = apply_T(T_pcd, world_pts).astype(np.float32)

        groups_map: Dict[Tuple[str, str], SemanticGroup] = {}
        total_masks_used = 0

        print("\nProcessing frames...")

        for i, idx in enumerate(frame_ids):
            w, h, fx, fy, cx, cy = load_intrinsics_txt(intr_files[idx])
            T_cw = T_cw_list[idx]

            depth_raw = None
            if args.use_depth_consistency and depth_files:
                depth_raw = load_depth(depth_files[idx])

            frame_cache = build_frame_cache(
                world_pts=world_pts,
                T_cw=T_cw,
                w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                depth_raw=depth_raw,
                depth_scale=float(args.depth_scale),
                use_depth_consistency=bool(args.use_depth_consistency),
                depth_tau_min=float(args.depth_tau_min),
                depth_tau_k=float(args.depth_tau_k),
                depth_tau_front=float(args.depth_tau_front),
                voxel_size=voxel_size,
            )

            if frame_cache.world_idx.size == 0:
                if args.log_every > 0 and i % args.log_every == 0:
                    print(f"  Frame {idx:04d} ({i+1}/{len(frame_ids)}): no visible points, skip")
                continue

            fdir = get_frame_dir(masks_root, idx)
            mask_paths = sorted(
                [p for p in fdir.iterdir()
                 if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")],
                key=lambda p: natural_key(p.name)
            )

            aff_union: Optional[np.ndarray] = None
            for mp2 in mask_paths:
                typ_tmp, _, _ = parse_mask_type_name_inst(mp2.stem)
                if typ_tmp != "affordance":
                    continue
                try:
                    m_tmp = load_mask(mp2)
                except Exception:
                    continue
                if m_tmp.sum() == 0:
                    continue
                mb = (m_tmp > 0)
                aff_union = mb if aff_union is None else (aff_union | mb)

            used_this_frame = 0

            for mp in mask_paths:
                typ, raw_name, inst_id = parse_mask_type_name_inst(mp.stem)
                if typ not in ("object", "affordance"):
                    continue

                try:
                    mask01 = load_mask(mp)
                except Exception as e:
                    print(f"  Warning: failed to load {mp.name}: {e}")
                    continue

                if mask01.sum() == 0:
                    continue

                if typ == "object" and aff_union is not None:
                    mask01 = subtract_union_mask(mask01, aff_union)
                    if mask01.sum() == 0:
                        continue

                cname = canon_name(raw_name, name_map)

                mask_visible_idx, fg_idx = query_mask_from_cache(
                    frame_cache, mask01, bool(args.use_depth_consistency)
                )

                if mask_visible_idx is None or len(mask_visible_idx) == 0:
                    continue

                key = (typ, cname)
                if key not in groups_map:
                    groups_map[key] = SemanticGroup(typ=typ, name=cname)

                groups_map[key].add_obs(
                    raw_name=raw_name,
                    mask_name=mp.name,
                    visible_indices=mask_visible_idx,
                    fg_indices=fg_idx,
                    max_sample_masks=10,
                )
                used_this_frame += 1

            total_masks_used += used_this_frame
            if args.log_every > 0 and (i % args.log_every == 0 or i == len(frame_ids) - 1):
                print(f"  Frame {idx:04d} ({i+1}/{len(frame_ids)}): "
                      f"{used_this_frame} masks, groups={len(groups_map)}")

        groups_list = list(groups_map.values())
        print(f"\nTotal masks used: {total_masks_used}")
        print(f"Semantic groups:  {len(groups_list)}")

        nodes_with_indices: List[Tuple[dict, set]] = []

        for g in groups_list:
            n_views = max(1, g.observations)

            if g.typ == "affordance":
                min_vis_base = int(args.aff_min_vis) if args.aff_min_vis >= 0 else int(args.min_vis)
                min_vis_frac = float(args.aff_min_vis_frac) if args.aff_min_vis_frac >= 0 else float(args.min_vis_frac)
                min_fg = int(args.aff_min_fg) if args.aff_min_fg >= 0 else int(args.min_fg)
                min_fg_ratio = float(args.aff_min_fg_ratio) if args.aff_min_fg_ratio >= 0 else float(args.min_fg_ratio)
                use_w = bool(args.aff_use_wilson)
                w_p0 = float(args.aff_wilson_p0)
                w_z = float(args.aff_wilson_z)
            else:
                min_vis_base = int(args.min_vis)
                min_vis_frac = float(args.min_vis_frac)
                min_fg = int(args.min_fg)
                min_fg_ratio = float(args.min_fg_ratio)
                use_w = bool(args.use_wilson)
                w_p0 = float(args.wilson_p0)
                w_z = float(args.wilson_z)

            min_vis_eff = int(max(min_vis_base, math.ceil(min_vis_frac * n_views)))

            kept = filter_kept_vectorized(g, min_vis_eff, min_fg, use_w, w_p0, w_z, min_fg_ratio)

            if not kept:
                print(f"[GROUP] {g.name}  obs={g.observations}  candidates={len(g.indices)}  kept=0")
                continue

            eps_here = aff_dbscan_eps if g.typ == "affordance" else obj_dbscan_eps
            instance_clusters = split_into_instances_by_dbscan(
                world_pts, kept, float(eps_here), int(args.dbscan_min_samples)
            )
            print(f"[GROUP] {g.name}  obs={g.observations}  candidates={len(g.indices)}  "
                  f"kept={len(kept)}  instances={len(instance_clusters)}")

            for inst_i, idx_filtered in enumerate(instance_clusters):
                if not idx_filtered:
                    continue
                idx_arr = np.array(sorted(map(int, idx_filtered)), dtype=np.int64)
                pts_sel = world_pts[idx_arr].astype(np.float32)
                if pts_sel.shape[0] == 0:
                    continue

                if args.use_dbscan_position:
                    c_db, dbinfo = dbscan_largest_cluster_centroid(
                        pts_sel, float(eps_here), int(args.dbscan_min_samples))
                    c_world = c_db if c_db is not None else pts_sel.mean(axis=0)
                else:
                    c_world = pts_sel.mean(axis=0)
                    dbinfo = None

                if c_world is None or not np.isfinite(np.asarray(c_world)).all():
                    continue

                c_out = apply_T(T_out, np.asarray(c_world).reshape(1, 3)).reshape(3)
                aabb = compute_aabb_from_points(pts_sel)

                node = {
                    "type": g.typ,
                    "name": g.name,
                    "inst_id": str(inst_i).zfill(3),
                    "names": g.all_names(),
                    "position": c_out.tolist(),
                    "confidence": {
                        "points_final": int(len(idx_filtered)),
                        "observations": int(g.observations)
                    },
                    "debug": {
                        "sample_masks": list(g.sample_masks),
                        "dbscan_position": dbinfo,
                        "min_vis_eff": int(min_vis_eff),
                        "bbox_world": {
                            "min": aabb[0].tolist(),
                            "max": aabb[1].tolist(),
                        } if aabb is not None else None,
                    },
                }
                nodes_with_indices.append((node, set(idx_filtered)))

        merge_info_all = {}

        if args.merge_nodes:
            nodes_with_indices, merge_info = merge_nodes_by_index_overlap(
                nodes_with_indices, world_pts, T_out,
                bool(args.use_dbscan_position), obj_dbscan_eps, aff_dbscan_eps,
                int(args.dbscan_min_samples), int(args.merge_min_common), float(args.merge_min_ratio),
            )
            print(f"[MergeNodes-Index] before={merge_info['nodes_before']} after={merge_info['nodes_after']} "
                  f"merged={merge_info['merged_components']}")
            merge_info_all["index_overlap_merge"] = merge_info
        else:
            merge_info_all["index_overlap_merge"] = {"enabled": False}

        if args.merge_object_bbox:
            nodes_with_indices, bbox_merge_info = merge_object_nodes_by_bbox(
                nodes_with_indices=nodes_with_indices,
                world_pts=world_pts,
                T_out=T_out,
                use_dbscan_position=bool(args.use_dbscan_position),
                obj_dbscan_eps=float(obj_dbscan_eps),
                dbscan_min_samples=int(args.dbscan_min_samples),
                bbox_iou_thr=float(args.bbox_iou_thr),
                bbox_contain_thr=float(args.bbox_contain_thr),
            )
            print(f"[MergeNodes-BBox-ObjectOnly] before={bbox_merge_info['nodes_before']} "
                  f"after={bbox_merge_info['nodes_after']} "
                  f"merged={bbox_merge_info['merged_components']}")
            merge_info_all["object_bbox_merge"] = bbox_merge_info
        else:
            merge_info_all["object_bbox_merge"] = {"enabled": False}

        nodes_with_indices, own_info = enforce_affordance_priority_ownership(
            nodes_with_indices, world_pts, T_out,
            bool(args.use_dbscan_position), float(obj_dbscan_eps), int(args.dbscan_min_samples),
        )
        print(f"[Ownership-AffordancePriority] removed_points={own_info['removed_points']} "
              f"dropped_obj={own_info['dropped_object_nodes']}")

        nodes_with_indices = sorted(
            nodes_with_indices,
            key=lambda x: (x[0]["type"], x[0]["name"], x[0].get("inst_id", ""),
                           *x[0]["position"])
        )

        nodes: List[dict] = []
        node_indices_rows: List[dict] = []
        for nid, (n, idxset) in enumerate(nodes_with_indices, start=1):
            n["id"] = int(nid)
            rgb = color_for_node_key(n.get("type", ""), n.get("name", ""), n.get("inst_id", ""))
            n["color_rgb01"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
            nodes.append(n)
            node_indices_rows.append({
                "id": int(nid),
                "type": n.get("type"),
                "name": n.get("name"),
                "inst_id": n.get("inst_id", ""),
                "point_indices": sorted(map(int, idxset)),
            })

        nodes = assign_affordances_to_nearest_object(nodes)

        objects = [n for n in nodes if n["type"] == "object"]
        affords = [n for n in nodes if n["type"] == "affordance"]
        print(f"Objects: {len(objects)}, Affordances: {len(affords)}")

        pts_nodes = np.array([n["position"] for n in nodes], dtype=np.float32)
        cols_nodes = np.array([n.get("color_rgb01", [1.0, 1.0, 1.0]) for n in nodes], dtype=np.float64)
        nodes_positions_ply = out_path.parent / "nodes_positions.ply"
        save_xyz_as_ply(pts_nodes, nodes_positions_ply, colors_rgb01=cols_nodes)
        print(f"[NodesPCD] saved: {nodes_positions_ply} (N={len(pts_nodes)})")

        if args.save_ply:
            for ni, (node, idxset) in enumerate(nodes_with_indices, start=1):
                if not idxset:
                    continue
                idx_list = np.array(sorted(map(int, idxset)), dtype=np.int64)
                pts = world_pts[idx_list].astype(np.float32)
                if args.ply_apply_T_out:
                    pts = apply_T(T_out, pts).astype(np.float32)
                rgb = tuple(node.get("color_rgb01", [1.0, 1.0, 1.0]))
                fn = (f"node_{ni:04d}_{node['type']}_"
                      f"{safe_filename(node['name'])}__{node.get('inst_id','')}.ply")
                save_points_as_ply(pts, ply_dir / fn,
                                   color_rgb01=rgb, voxel_down=float(args.ply_voxel_down))

        output = {
            "scene_id": int(scene_id),
            "sequence_id": str(seq_id),
            "pcd_path": str(pcd_path),
            "transform_npy": str(transform_path) if transform_path else None,
            "name_map_json": str(name_map_path) if name_map_path else None,
            "nodes": nodes,
            "stats": {
                "frames_total": len(frame_ids_all),
                "frames_used": len(frame_ids),
                "masks_used": total_masks_used,
                "semantic_groups": len(groups_map),
                "objects": len(objects),
                "affordances": len(affords),
                "inst_splitting": "dbscan",
            },
            "params": {
                "voxel_cap": {"enabled": voxel_size > 0, "voxel_size_m": voxel_size},
                "use_depth_consistency": bool(args.use_depth_consistency),
                "depth_dir": str(depth_dir) if depth_dir else None,
                "depth_scale": float(args.depth_scale),
                "depth_tau_min": float(args.depth_tau_min),
                "depth_tau_k": float(args.depth_tau_k),
                "depth_tau_front": float(args.depth_tau_front),
                "use_wilson": bool(args.use_wilson),
                "wilson_p0": float(args.wilson_p0),
                "wilson_z": float(args.wilson_z),
                "min_vis": int(args.min_vis),
                "min_vis_frac": float(args.min_vis_frac),
                "min_fg": int(args.min_fg),
                "min_fg_ratio": float(args.min_fg_ratio),
                "use_dbscan_position": bool(args.use_dbscan_position),
                "dbscan_eps": float(args.dbscan_eps),
                "dbscan_min_samples": int(args.dbscan_min_samples),
                "save_ply": bool(args.save_ply),
                "ply_dir": str(ply_dir) if args.save_ply else None,
                "ply_apply_T_out": bool(args.ply_apply_T_out),
                "ply_voxel_down": float(args.ply_voxel_down),
                "merge_nodes": merge_info_all,
                "ownership": own_info,
            }
        }

        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        idx_out_path = out_path.with_name(out_path.stem + "_point_indices.json")
        idx_out_path.write_text(json.dumps({
            "scene_id": int(scene_id),
            "sequence_id": str(seq_id),
            "pcd_path": str(pcd_path),
            "node_point_indices": node_indices_rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save experiment configuration
        exp_config = create_build_scene_graph_config(
            args, scene_id, seq_id, pcd_path, transform_path, name_map_path, depth_dir
        )
        exp_config['results'] = {
            'frames_total': len(frame_ids_all),
            'frames_used': len(frame_ids),
            'masks_used': total_masks_used,
            'semantic_groups': len(groups_map),
            'objects_count': len(objects),
            'affordances_count': len(affords),
            'total_nodes': len(nodes),
        }
        save_build_scene_graph_config(exp_config, str(out_path.parent))

        print(f"\n{'='*60}")
        print(f"Scene graph  →  {out_path}")
        print(f"Point indices → {idx_out_path}")
        if args.save_ply:
            print(f"PLY files    →  {ply_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()