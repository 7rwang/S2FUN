#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python refine_masks.py \
  --scene_id 421393 \
  --masks_root /nas/qirui/sam3/scenefun3d_ex/experiments4sam3_mask/all_scenes/421393/masks \
  --data_root /nas/qirui/scenefun3d/val \
  --out_root /nas/qirui/sam3/scenefun3d_ex/refined_masks_3d/421393 \
  --depth_scale 1000 \
  --voxel_size 0.1 \
  --contain_thr 0.6 \
  --suppress_thr_handle 0.9 \
  --save_projected_ply
'''
import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Utils
# =========================================================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).astype(np.float64)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=np.float64)
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
                raise ValueError(f"{traj_path} line {line_no}: expect 7 fields, got {len(parts)}")
            _, ax, ay, az, tx, ty, tz = map(float, parts)
            R = rodrigues(np.array([ax, ay, az], dtype=np.float64))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            T_list.append(T)
    if not T_list:
        raise ValueError(f"No valid poses found in {traj_path}")
    return T_list


def load_intrinsics_txt(p: Path) -> Tuple[int, int, float, float, float, float]:
    vals = p.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Intrinsics {p}: expect 6 values, got {len(vals)}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:])
    return w, h, fx, fy, cx, cy


def load_mask(mask_path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required.")
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


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

    raise RuntimeError("Need cv2 or .npy depth files.")


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


def find_depth_dir_auto(seq_dir: Path) -> Path:
    for cand in ["hires_wide_depth", "hires_depth", "depth", "depths"]:
        p = seq_dir / cand
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"No depth dir found under: {seq_dir}")


def list_depth_files(depth_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy")
    files = [p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))


def list_intr_files(intr_dir: Path) -> List[Path]:
    return sorted([p for p in intr_dir.iterdir() if p.is_file()], key=lambda p: natural_key(p.name))


def list_frame_dirs(masks_root: Path) -> List[int]:
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")
    sub = [p for p in masks_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if not sub:
        return [0]
    return sorted(int(p.name) for p in sub)


def get_frame_dir(masks_root: Path, idx: int) -> Path:
    p = masks_root / str(idx)
    return p if p.is_dir() else masks_root


def parse_mask_type_name_inst(stem: str) -> Tuple[str, str, str]:
    s = stem.strip()
    if s.startswith("CTX"):
        typ, prefix = "object", "CTX"
    elif s.startswith("INT"):
        typ, prefix = "affordance", "INT"
    else:
        typ, prefix = "object", None

    parts = [p for p in (s.split("__") if "__" in s else s.split("_")) if p]

    if prefix and parts and parts[0] == prefix:
        parts = parts[1:]
    elif prefix and parts and parts[0].startswith(prefix):
        parts[0] = parts[0].replace(prefix, "", 1)

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


# =========================================================
# Semantic handling
# =========================================================

def handle_group_name(name: str) -> str:
    n = normalize_name(name)
    if n in {"door handle", "rotate door handle"}:
        return "door_handle_group"
    return n


def same_semantic_group(name_a: str, name_b: str) -> bool:
    return handle_group_name(name_a) == handle_group_name(name_b)


def is_handle_like(name: str) -> bool:
    return handle_group_name(name) == "door_handle_group"


# =========================================================
# 2D -> 3D
# =========================================================

def mask_depth_filter(depth_vals: np.ndarray, tau_k: float = 3.0, tau_min: float = 0.03) -> np.ndarray:
    if depth_vals.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(depth_vals))
    mad = float(np.median(np.abs(depth_vals - med)))
    tau = max(tau_min, tau_k * 1.4826 * mad)
    return np.abs(depth_vals - med) <= tau


def backproject_mask_to_world(
    mask01: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    intr: Tuple[int, int, float, float, float, float],
    Twc: np.ndarray,
    use_depth_filter: bool = True,
) -> np.ndarray:
    h_img, w_img = mask01.shape[:2]
    w0, h0, fx, fy, cx, cy = intr
    if (w_img != w0) or (h_img != h0):
        raise ValueError(f"Mask shape {mask01.shape} != intrinsics size {(h0, w0)}")

    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_raw[ys, xs].astype(np.float32) / float(depth_scale)
    valid = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if use_depth_filter:
        keep = mask_depth_filter(z)
        xs, ys, z = xs[keep], ys[keep], z[keep]
        if z.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

    x_cam = (xs.astype(np.float32) - cx) * z / fx
    y_cam = (ys.astype(np.float32) - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)

    T_use = np.linalg.inv(Twc)
    R = T_use[:3, :3].astype(np.float32)
    t = T_use[:3, 3].astype(np.float32)
    pts_world = (R @ pts_cam.T).T + t[None, :]
    return pts_world


def voxel_set_from_points(points_xyz: np.ndarray, voxel_size: float) -> Set[Tuple[int, int, int]]:
    if points_xyz.size == 0:
        return set()
    vox = np.floor(points_xyz / voxel_size).astype(np.int64)
    return set(map(tuple, vox.tolist()))


def contain_small(a: Set[Tuple[int, int, int]], b: Set[Tuple[int, int, int]]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(min(len(a), len(b)), 1)


# =========================================================
# PLY save
# =========================================================

def save_xyz_as_ply(points_xyz: np.ndarray, out_path: Path, voxel_down: float = 0.0):
    if points_xyz is None or points_xyz.size == 0:
        return
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("Saving PLY requires open3d.") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))

    if voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(voxel_down))

    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)


# =========================================================
# Data structures
# =========================================================

@dataclass
class MaskObs:
    obs_id: int
    frame_idx: int
    src_path: str
    src_name: str
    rel_src_path: str
    mask_name: str
    mask_group: str
    mask_type: str
    raw_local_id: str
    area_2d: int
    voxel_count: int
    voxels: Set[Tuple[int, int, int]]
    points_world_count: int
    projected_ply_relpath: Optional[str] = None

    assigned_instance_id: Optional[int] = None
    dropped: bool = False
    dropped_by_obs_id: Optional[int] = None
    dropped_by_instance_id: Optional[int] = None


@dataclass
class InstanceState:
    instance_id: int
    semantic_group: str
    observations: List[int] = field(default_factory=list)
    union_voxels: Set[Tuple[int, int, int]] = field(default_factory=set)
    rep_voxels: Set[Tuple[int, int, int]] = field(default_factory=set)
    rep_area_2d: int = 0
    rep_obs_id: Optional[int] = None
    frames: Set[int] = field(default_factory=set)


# =========================================================
# Core logic
# =========================================================

def choose_best_instance(
    obs: MaskObs,
    instances: List[InstanceState],
    contain_thr: float
) -> Optional[int]:
    best_inst_id = None
    best_score = -1.0

    for inst in instances:
        if inst.semantic_group != obs.mask_group:
            continue
        ref_voxels = inst.rep_voxels if inst.rep_voxels else inst.union_voxels
        score = contain_small(obs.voxels, ref_voxels)
        if score > best_score:
            best_score = score
            best_inst_id = inst.instance_id

    if best_score >= contain_thr:
        return best_inst_id
    return None


def update_instance_with_obs(inst: InstanceState, obs: MaskObs):
    inst.observations.append(obs.obs_id)
    inst.union_voxels |= obs.voxels
    inst.frames.add(obs.frame_idx)

    if inst.rep_obs_id is None:
        inst.rep_obs_id = obs.obs_id
        inst.rep_voxels = set(obs.voxels)
        inst.rep_area_2d = obs.area_2d
        return

    if inst.semantic_group == "door_handle_group":
        if obs.area_2d < inst.rep_area_2d:
            inst.rep_obs_id = obs.obs_id
            inst.rep_voxels = set(obs.voxels)
            inst.rep_area_2d = obs.area_2d


def incremental_assign_instances(
    obs_list: List[MaskObs],
    contain_thr: float,
    suppress_thr_handle: float,
) -> Tuple[List[MaskObs], List[InstanceState]]:
    instances: List[InstanceState] = []
    next_instance_id = 1
    obs_id_to_obs = {o.obs_id: o for o in obs_list}

    obs_list_sorted = sorted(obs_list, key=lambda o: (o.frame_idx, natural_key(o.src_name)))

    iterator = obs_list_sorted
    if tqdm is not None:
        iterator = tqdm(obs_list_sorted, desc="Assigning instances", dynamic_ncols=True)

    for obs in iterator:
        matched_instance_id = choose_best_instance(obs, instances, contain_thr)

        if matched_instance_id is None:
            inst = InstanceState(
                instance_id=next_instance_id,
                semantic_group=obs.mask_group,
            )
            obs.assigned_instance_id = next_instance_id
            update_instance_with_obs(inst, obs)
            instances.append(inst)
            next_instance_id += 1
            continue

        obs.assigned_instance_id = matched_instance_id
        inst = next(x for x in instances if x.instance_id == matched_instance_id)

        if inst.semantic_group == "door_handle_group":
            overlap = contain_small(obs.voxels, inst.rep_voxels if inst.rep_voxels else inst.union_voxels)
            if overlap >= suppress_thr_handle and inst.rep_obs_id is not None:
                rep_obs = obs_id_to_obs[inst.rep_obs_id]

                if obs.area_2d < rep_obs.area_2d:
                    rep_obs.dropped = True
                    rep_obs.dropped_by_obs_id = obs.obs_id
                    rep_obs.dropped_by_instance_id = inst.instance_id

                    inst.rep_obs_id = obs.obs_id
                    inst.rep_voxels = set(obs.voxels)
                    inst.rep_area_2d = obs.area_2d
                    update_instance_with_obs(inst, obs)
                else:
                    obs.dropped = True
                    obs.dropped_by_obs_id = rep_obs.obs_id
                    obs.dropped_by_instance_id = inst.instance_id
            else:
                update_instance_with_obs(inst, obs)
        else:
            update_instance_with_obs(inst, obs)

    return obs_list_sorted, instances


# =========================================================
# IO
# =========================================================

def build_output_name(instance_id: int, src_name: str) -> str:
    return f"instance_{instance_id:04d}__{src_name}"


def save_outputs(
    obs_list: List[MaskObs],
    instances: List[InstanceState],
    out_root: Path,
    save_dropped_masks: bool,
    last_frame_idx: int,
):
    masks_out = out_root / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    dropped_out = out_root / "dropped_masks"
    if save_dropped_masks:
        dropped_out.mkdir(parents=True, exist_ok=True)

    # 先把 0000 ~ last_frame_idx 的目录全部建出来
    for frame_idx in range(last_frame_idx + 1):
        (masks_out / f"{frame_idx:04d}").mkdir(parents=True, exist_ok=True)
        if save_dropped_masks:
            (dropped_out / f"{frame_idx:04d}").mkdir(parents=True, exist_ok=True)

    kept_records = []
    dropped_records = []

    for obs in obs_list:
        src = Path(obs.src_path)
        out_name = build_output_name(obs.assigned_instance_id, obs.src_name)
        frame_dir = f"{obs.frame_idx:04d}"

        if not obs.dropped:
            dst_dir = masks_out / frame_dir
            shutil.copy2(src, dst_dir / out_name)
            kept_records.append({
                "obs_id": obs.obs_id,
                "instance_id": obs.assigned_instance_id,
                "frame_idx": obs.frame_idx,
                "mask_name": obs.mask_name,
                "mask_group": obs.mask_group,
                "mask_type": obs.mask_type,
                "orig_rel_path": obs.rel_src_path,
                "saved_rel_path": str(Path("masks") / frame_dir / out_name),
                "projected_ply_relpath": obs.projected_ply_relpath,
                "raw_local_id": obs.raw_local_id,
                "area_2d": obs.area_2d,
                "voxel_count": obs.voxel_count,
                "points_world_count": obs.points_world_count,
            })
        else:
            dropped_records.append({
                "obs_id": obs.obs_id,
                "instance_id": obs.assigned_instance_id,
                "frame_idx": obs.frame_idx,
                "mask_name": obs.mask_name,
                "mask_group": obs.mask_group,
                "mask_type": obs.mask_type,
                "orig_rel_path": obs.rel_src_path,
                "projected_ply_relpath": obs.projected_ply_relpath,
                "raw_local_id": obs.raw_local_id,
                "area_2d": obs.area_2d,
                "voxel_count": obs.voxel_count,
                "points_world_count": obs.points_world_count,
                "dropped_by_obs_id": obs.dropped_by_obs_id,
                "dropped_by_instance_id": obs.dropped_by_instance_id,
            })
            if save_dropped_masks:
                dst_dir = dropped_out / frame_dir
                shutil.copy2(src, dst_dir / out_name)

    instance_summary = []
    for inst in instances:
        surviving_obs = [o for o in obs_list if (o.assigned_instance_id == inst.instance_id and not o.dropped)]
        if not surviving_obs:
            continue
        instance_summary.append({
            "instance_id": inst.instance_id,
            "semantic_group": inst.semantic_group,
            "num_observations_total": len(inst.observations),
            "num_observations_kept": len(surviving_obs),
            "num_frames": len(set(o.frame_idx for o in surviving_obs)),
            "rep_obs_id": next((o.obs_id for o in surviving_obs if o.obs_id == inst.rep_obs_id), None),
        })

    with (out_root / "instance_map.json").open("w", encoding="utf-8") as f:
        json.dump({
            "kept": kept_records,
            "dropped": dropped_records,
            "instances": instance_summary,
            "last_frame_idx": int(last_frame_idx),
        }, f, indent=2, ensure_ascii=False)


# =========================================================
# Load observations from scene
# =========================================================

def load_observations_from_scene(
    scene_id: int,
    masks_root: Path,
    data_root: Path,
    depth_scale: float,
    voxel_size: float,
    use_depth_filter: bool,
    min_points_per_mask: int,
    save_projected_ply: bool,
    projected_ply_root: Optional[Path],
    projected_ply_voxel_down: float,
) -> Tuple[List[MaskObs], Dict]:
    scene_root = data_root / str(scene_id)
    seq_dir = find_first_sequence_dir(scene_root)

    traj_path = find_traj_path(seq_dir)
    intr_dir = find_intr_dir(seq_dir)
    depth_dir = find_depth_dir_auto(seq_dir)

    Twc_list = read_traj_Twc(traj_path)
    intr_files = list_intr_files(intr_dir)
    depth_files = list_depth_files(depth_dir)

    n_frames_available = min(len(Twc_list), len(intr_files), len(depth_files))
    frame_indices = list_frame_dirs(masks_root)

    obs_list: List[MaskObs] = []
    next_obs_id = 0
    skipped_empty = 0
    skipped_too_few_points = 0

    frame_iter = frame_indices
    if tqdm is not None:
        frame_iter = tqdm(frame_indices, desc="Loading + backprojecting", dynamic_ncols=True)

    for frame_idx in frame_iter:
        if frame_idx >= n_frames_available:
            continue

        frame_dir = get_frame_dir(masks_root, frame_idx)
        mask_files = sorted(
            [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
            key=lambda p: natural_key(p.name)
        )
        if not mask_files:
            continue

        intr = load_intrinsics_txt(intr_files[frame_idx])
        depth = load_depth(depth_files[frame_idx])
        Twc = Twc_list[frame_idx]

        for mask_path in mask_files:
            typ, name, raw_local_id = parse_mask_type_name_inst(mask_path.stem)
            mask01 = load_mask(mask_path)
            area_2d = int(mask01.sum())
            pts_world = backproject_mask_to_world(
                mask01=mask01,
                depth_raw=depth,
                depth_scale=depth_scale,
                intr=intr,
                Twc=Twc,
                use_depth_filter=use_depth_filter,
            )

            if pts_world.shape[0] == 0:
                skipped_empty += 1
                continue
            if pts_world.shape[0] < min_points_per_mask:
                skipped_too_few_points += 1
                continue

            voxels = voxel_set_from_points(pts_world, voxel_size)

            projected_ply_relpath = None
            if save_projected_ply and projected_ply_root is not None:
                ply_name = f"obs_{next_obs_id:06d}__{mask_path.stem}.ply"
                ply_path = projected_ply_root / f"{frame_idx:04d}" / ply_name
                save_xyz_as_ply(pts_world, ply_path, voxel_down=projected_ply_voxel_down)
                projected_ply_relpath = str(Path("projected_ply") / f"{frame_idx:04d}" / ply_name)

            obs_list.append(MaskObs(
                obs_id=next_obs_id,
                frame_idx=frame_idx,
                src_path=str(mask_path),
                src_name=mask_path.name,
                rel_src_path=str(mask_path.relative_to(masks_root)),
                mask_name=name,
                mask_group=handle_group_name(name),
                mask_type=typ,
                raw_local_id=raw_local_id,
                area_2d=area_2d,
                voxel_count=len(voxels),
                voxels=voxels,
                points_world_count=int(pts_world.shape[0]),
                projected_ply_relpath=projected_ply_relpath,
            ))
            next_obs_id += 1

    meta = {
        "scene_id": scene_id,
        "num_observations_loaded": len(obs_list),
        "skipped_empty": skipped_empty,
        "skipped_too_few_points": skipped_too_few_points,
        "last_frame_idx": int(n_frames_available - 1),
    }
    return obs_list, meta


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Incremental instance assignment from 2D mask + depth -> 3D overlap. "
            "Only door handle / rotate door handle use keep-smaller logic. "
            "Saved file names include instance id directly."
        )
    )
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--masks_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--voxel_size", type=float, default=0.01)
    ap.add_argument("--use_depth_filter", action="store_true")
    ap.add_argument("--min_points_per_mask", type=int, default=20)

    # single threshold for "same object"
    ap.add_argument("--contain_thr", type=float, default=0.8)

    # handle-only suppress threshold
    ap.add_argument("--suppress_thr_handle", type=float, default=0.9)

    ap.add_argument("--save_dropped_masks", action="store_true")

    # new
    ap.add_argument("--save_projected_ply", action="store_true")
    ap.add_argument("--projected_ply_voxel_down", type=float, default=0.0)

    args = ap.parse_args()

    masks_root = Path(args.masks_root)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    projected_ply_root = None
    if args.save_projected_ply:
        projected_ply_root = out_root / "projected_ply"
        projected_ply_root.mkdir(parents=True, exist_ok=True)

    obs_list, meta = load_observations_from_scene(
        scene_id=args.scene_id,
        masks_root=masks_root,
        data_root=data_root,
        depth_scale=args.depth_scale,
        voxel_size=args.voxel_size,
        use_depth_filter=args.use_depth_filter,
        min_points_per_mask=args.min_points_per_mask,
        save_projected_ply=args.save_projected_ply,
        projected_ply_root=projected_ply_root,
        projected_ply_voxel_down=args.projected_ply_voxel_down,
    )

    if not obs_list:
        print("No valid mask observations found.")
        return

    obs_list, instances = incremental_assign_instances(
        obs_list=obs_list,
        contain_thr=args.contain_thr,
        suppress_thr_handle=args.suppress_thr_handle,
    )

    save_outputs(
        obs_list=obs_list,
        instances=instances,
        out_root=out_root,
        save_dropped_masks=args.save_dropped_masks,
        last_frame_idx=meta["last_frame_idx"],
    )

    num_kept = sum(1 for x in obs_list if not x.dropped)
    num_dropped = sum(1 for x in obs_list if x.dropped)
    num_instances = len({x.assigned_instance_id for x in obs_list if x.assigned_instance_id is not None})

    with (out_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            **meta,
            "contain_thr": args.contain_thr,
            "suppress_thr_handle": args.suppress_thr_handle,
            "voxel_size": args.voxel_size,
            "num_instances": num_instances,
            "num_kept": num_kept,
            "num_dropped": num_dropped,
            "save_projected_ply": bool(args.save_projected_ply),
            "projected_ply_voxel_down": args.projected_ply_voxel_down,
            "naming_rule": "saved filename = instance_{instance_id:04d}__original_filename",
        }, f, indent=2, ensure_ascii=False)

    print(f"Done. Results saved to: {out_root}")
    print(f"Loaded observations: {len(obs_list)}")
    print(f"Instances: {num_instances}")
    print(f"Kept: {num_kept}")
    print(f"Dropped: {num_dropped}")


if __name__ == "__main__":
    main()