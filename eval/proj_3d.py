import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# -------------------------
# Small utils
# -------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def load_intrinsics_txt(p: Path) -> Tuple[int, int, float, float, float, float]:
    """width height fx fy cx cy"""
    vals = p.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Intrinsics {p}: expect 6 values, got {len(vals)}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal: fx={fx}, fy={fy}")
    return w, h, fx, fy, cx, cy


def load_mask(mask_path: Path) -> np.ndarray:
    """binary uint8 (H,W)"""
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install: pip install opencv-python")
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def load_depth_png(depth_path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV not available")

    depth_path = Path(depth_path)

    if not depth_path.exists():
        raise FileNotFoundError(f"[DEPTH] file not exists: {depth_path}")

    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if d is None:
        raise RuntimeError(
            f"[DEPTH] cv2.imread returned None\n"
            f"  path: {depth_path}\n"
            f"  exists: {depth_path.exists()}\n"
            f"  size: {depth_path.stat().st_size if depth_path.exists() else 'NA'}"
        )

    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)



# -------------------------
# Pose reading (traj)
# -------------------------

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).astype(np.float64)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def validate_transform(T: np.ndarray, name: str = "Transform") -> None:
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(det) < 0.01:
        raise ValueError(f"{name} det too small: {det}")
    should_be_I = R.T @ R
    err = float(np.max(np.abs(should_be_I - np.eye(3))))
    if err > 0.1:
        raise ValueError(f"{name} rotation not orthogonal, max err={err}")


def read_traj_Twc(traj_path: Path) -> List[np.ndarray]:
    """
    traj line: timestamp ax ay az tx ty tz
    returns list of T_wc (world <- camera)
    """
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
            rvec = np.array([ax, ay, az], dtype=np.float64)
            R = rodrigues(rvec)
            t = np.array([tx, ty, tz], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            validate_transform(T, f"traj line {line_no}")
            T_list.append(T)
    if not T_list:
        raise ValueError(f"No valid poses in: {traj_path}")
    return T_list


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# -------------------------
# Core: mask+depth -> 3D points
# -------------------------

@dataclass
class Intrinsics:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float


def backproject_mask_to_camera_points(
    depth_raw: np.ndarray,
    mask01: np.ndarray,
    K: Intrinsics,
    depth_scale: float = 1000.0,
    depth_min: float = 1e-6,
    depth_max: float = 1e6,
    stride: int = 1,
) -> np.ndarray:
    if depth_raw.shape[:2] != (K.h, K.w):
        raise ValueError(f"depth shape {depth_raw.shape} != intrinsics {K.w}x{K.h}")
    if mask01.shape[:2] != (K.h, K.w):
        raise ValueError(f"mask shape {mask01.shape} != intrinsics {K.w}x{K.h}")

    m = (mask01 > 0)

    if stride > 1:
        mm = np.zeros_like(m, dtype=bool)
        mm[::stride, ::stride] = m[::stride, ::stride]
        m = mm

    vv, uu = np.nonzero(m)
    if vv.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_raw[vv, uu].astype(np.float32) / float(depth_scale)
    valid = np.isfinite(z) & (z > float(depth_min)) & (z < float(depth_max))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    vv = vv[valid].astype(np.float32)
    uu = uu[valid].astype(np.float32)
    z = z[valid]

    x = (uu - float(K.cx)) * z / float(K.fx)
    y = (vv - float(K.cy)) * z / float(K.fy)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3).astype(np.float32)
    pts64 = np.asarray(pts, dtype=np.float64)
    out = (T[:3, :3] @ pts64.T + T[:3, 3:4]).T
    return out.astype(np.float32)


# -------------------------
# Scene folder discovery
# -------------------------

def find_first_sequence_dir(scene_root: Path) -> Path:
    seq_dirs = [p for p in scene_root.iterdir() if p.is_dir()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence dirs under: {scene_root}")
    numeric = [p for p in seq_dirs if re.fullmatch(r"\d+", p.name)]
    if numeric:
        return sorted(numeric, key=lambda p: natural_key(p.name))[0]
    return sorted(seq_dirs, key=lambda p: natural_key(p.name))[0]


def find_intr_dir(seq_dir: Path, intr_dirname: str = "hires_wide_intrinsics") -> Path:
    p = seq_dir / intr_dirname
    if not p.exists():
        raise FileNotFoundError(f"Intrinsics dir not found: {p}")
    return p


def find_traj_path(seq_dir: Path, traj_name: str = "hires_poses.traj") -> Path:
    p = seq_dir / traj_name
    if not p.exists():
        raise FileNotFoundError(f"Trajectory not found: {p}")
    return p


def list_depth_pngs(depth_dir: Path) -> List[Path]:
    files = [p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(files, key=lambda p: natural_key(p.name))


# -------------------------
# Mask discovery
# -------------------------

def parse_annot_id_from_mask_name(p: Path) -> Optional[str]:
    aid = p.stem.strip()
    return aid if aid else None


def collect_masks_by_annot_id(masks_root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    masks_root = Path(masks_root)
    annot2items: Dict[str, List[Tuple[int, Path]]] = {}

    frame_dirs = [p for p in masks_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if frame_dirs:
        for fdir in sorted(frame_dirs, key=lambda p: int(p.name)):
            frame_idx = int(fdir.name)
            for mp in sorted(fdir.iterdir(), key=lambda p: natural_key(p.name)):
                if mp.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                aid = parse_annot_id_from_mask_name(mp)
                if aid is None:
                    continue
                annot2items.setdefault(aid, []).append((frame_idx, mp))
        return annot2items

    for mp in sorted(masks_root.iterdir(), key=lambda p: natural_key(p.name)):
        if mp.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        aid = parse_annot_id_from_mask_name(mp)
        if aid is None:
            continue
        m = re.search(r"frame(\d+)", mp.stem)
        frame_idx = int(m.group(1)) if m else 0
        annot2items.setdefault(aid, []).append((frame_idx, mp))

    return annot2items


# -------------------------
# PLY writing (RED POINT CLOUD)
# -------------------------

def write_ply_xyz_rgb_red(path: Path, xyz: np.ndarray) -> None:
    """Write Nx3 points to ASCII PLY with red color."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = int(xyz.shape[0])

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 0 0\n")


# -------------------------
# Public API (NO CLI)
# -------------------------

def extract_annot_pointclouds(
    *,
    data_root: Union[str, Path],
    scene_id: int,
    masks_root: Union[str, Path],
    out_dir: Union[str, Path],
    depth_scale: float = 1000.0,
    stride_pixels: int = 1,
    traj_is_T_wc: bool = True,
    intr_dirname: str = "hires_wide_intrinsics",
    traj_name: str = "hires_poses.traj",
    save_meta: bool = True,
    cache_depth: bool = True,
) -> Dict[str, Any]:

    data_root = Path(data_root)
    masks_root = Path(masks_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_root = data_root / str(scene_id)
    seq_dir = find_first_sequence_dir(scene_root)

    intr_dir = find_intr_dir(seq_dir, intr_dirname)
    traj_path = find_traj_path(seq_dir, traj_name)
    T_list = read_traj_Twc(traj_path)
    if not traj_is_T_wc:
        T_list = [invert_T(T) for T in T_list]

    intr_files = sorted(intr_dir.iterdir(), key=lambda p: natural_key(p.name))
    depth_dir = seq_dir / "hires_depth"
    depth_files = list_depth_pngs(depth_dir)

    annot2items = collect_masks_by_annot_id(masks_root)

    intr_cache: Dict[int, Intrinsics] = {}
    depth_cache: Dict[int, np.ndarray] = {}

    per_annot = {}

    for annot_id in sorted(annot2items.keys(), key=natural_key):
        all_pts = []

        for frame_idx, mp in annot2items[annot_id]:
            if frame_idx not in intr_cache:
                w, h, fx, fy, cx, cy = load_intrinsics_txt(intr_files[frame_idx])
                intr_cache[frame_idx] = Intrinsics(w, h, fx, fy, cx, cy)

            if cache_depth and frame_idx not in depth_cache:
                depth_cache[frame_idx] = load_depth_png(depth_files[frame_idx])

            K = intr_cache[frame_idx]
            depth_raw = depth_cache[frame_idx]
            mask01 = load_mask(mp)

            pts_c = backproject_mask_to_camera_points(
                depth_raw, mask01, K,
                depth_scale=depth_scale,
                stride=stride_pixels,
            )
            if pts_c.shape[0] == 0:
                continue

            pts_w = apply_T(T_list[frame_idx], pts_c)
            all_pts.append(pts_w)

        if not all_pts:
            continue

        xyz = np.concatenate(all_pts, axis=0)

        np.savez_compressed(out_dir / f"{annot_id}.npz", xyz=xyz)
        write_ply_xyz_rgb_red(out_dir / f"{annot_id}.ply", xyz)

    return {"status": "ok"}
