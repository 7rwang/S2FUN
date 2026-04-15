# -*- coding: utf-8 -*-

import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# -------------------------
# Transforms
# -------------------------

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix."""
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
    """Validate that T is a valid 4x4 transformation matrix."""
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")

    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(det) < 0.01:
        raise ValueError(f"{name} has near-zero determinant: {det}")

    should_be_I = R.T @ R
    error = np.max(np.abs(should_be_I - np.eye(3)))
    if error > 0.1:
        raise ValueError(f"{name} rotation not orthogonal, max error: {error}")


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


# -------------------------
# Voxel-based point cloud
# -------------------------

class VoxelCloud:
    """Voxel-based point cloud for deduplication + voxel-key IoU."""

    def __init__(self, voxel_size: float):
        self.v = float(voxel_size)
        self.keys: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _key(self, p: np.ndarray) -> Tuple[int, int, int]:
        v = self.v
        return (
            int(np.floor(p[0] / v)),
            int(np.floor(p[1] / v)),
            int(np.floor(p[2] / v)),
        )

    def add(self, pts: np.ndarray):
        if pts.size == 0:
            return
        for p in pts:
            if not np.isfinite(p).all():
                continue
            k = self._key(p)
            if k not in self.keys:
                self.keys[k] = p.astype(np.float32)

    def merge_from(self, other: "VoxelCloud"):
        for k, p in other.keys.items():
            if k not in self.keys:
                self.keys[k] = p

    def points(self) -> np.ndarray:
        if not self.keys:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(list(self.keys.values()), axis=0).astype(np.float32)

    def centroid(self) -> Optional[np.ndarray]:
        pts = self.points()
        if pts.shape[0] == 0:
            return None
        return pts.mean(axis=0)

    def size(self) -> int:
        return len(self.keys)

    def key_set(self) -> set:
        return set(self.keys.keys())


def voxel_iou(a: VoxelCloud, b: VoxelCloud) -> float:
    """
    IoU of voxel keys (intersection/union).
    Returns 0.0 if either is empty.
    """
    na = a.size()
    nb = b.size()
    if na == 0 or nb == 0:
        return 0.0
    sa = a.key_set()
    sb = b.key_set()
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    uni = len(sa) + len(sb) - inter
    return float(inter) / float(max(uni, 1))


# -------------------------
# Projection and selection (with depth-consistency)
# -------------------------

def project_and_select(
    world_pts: np.ndarray,
    T_cw: np.ndarray,
    w: int, h: int,
    fx: float, fy: float, cx: float, cy: float,
    mask01: np.ndarray,
    depth_raw: Optional[np.ndarray] = None,
    depth_scale: float = 1000.0,
    use_depth_consistency: bool = False,
    depth_tau_min: float = 0.05,
    depth_tau_k: float = 3.0,
    depth_tau_front: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world points to camera frame and select by mask (z-buffer).
    Optionally filter by depth consistency using depth map + MAD threshold.

    Returns:
      pts_selected: (M, 3) selected WORLD points
      idx_selected: (M,) int64 indices into the original world_pts array
    """
    if world_pts.size == 0:
        return world_pts.reshape(0, 3)

    if mask01.shape[0] != h or mask01.shape[1] != w:
        raise ValueError(
            f"Mask size {mask01.shape[1]}x{mask01.shape[0]} != intrinsics {w}x{h}"
        )

    if use_depth_consistency:
        if depth_raw is None:
            raise ValueError("use_depth_consistency=True but depth_raw is None")
        if depth_raw.shape[0] != h or depth_raw.shape[1] != w:
            raise ValueError(
                f"Depth size {depth_raw.shape[1]}x{depth_raw.shape[0]} != intrinsics {w}x{h}"
            )

    # world -> camera
    pts_cam = apply_T(T_cw, world_pts)

    z = pts_cam[:, 2]
    valid = (z > 0) & np.isfinite(z) & np.isfinite(pts_cam).all(axis=1)
    if not np.any(valid):
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int64)
        return empty_pts, empty_idx

    pts_valid = pts_cam[valid]
    z_valid = z[valid].astype(np.float32)
    indices_valid = np.where(valid)[0]

    z_valid = np.maximum(z_valid, 1e-6)

    u = np.round(pts_valid[:, 0] * fx / z_valid + cx).astype(np.int32)
    v = np.round(pts_valid[:, 1] * fy / z_valid + cy).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int64)
        return empty_pts, empty_idx

    u = u[inside]
    v = v[inside]
    z_valid = z_valid[inside]
    indices_valid = indices_valid[inside]

    # z-buffer per pixel
    pixel_coords = v.astype(np.int64) * w + u.astype(np.int64)

    depth_order = np.argsort(z_valid)  # near first
    pixel_coords_sorted = pixel_coords[depth_order]
    indices_sorted = indices_valid[depth_order]
    u_sorted = u[depth_order]
    v_sorted = v[depth_order]
    z_sorted = z_valid[depth_order]

    _, unique_idx = np.unique(pixel_coords_sorted, return_index=True)

    u_final = u_sorted[unique_idx]
    v_final = v_sorted[unique_idx]
    world_indices_final = indices_sorted[unique_idx]
    z_pc_final = z_sorted[unique_idx]

    in_mask = mask01[v_final, u_final] > 0
    if not np.any(in_mask):
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int64)
        return empty_pts, empty_idx

    if use_depth_consistency:
        z_img = depth_raw[v_final, u_final].astype(np.float32) / float(depth_scale)
        valid_depth = np.isfinite(z_img) & (z_img > 0)

        keep0 = in_mask & valid_depth
        if not np.any(keep0):
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_idx = np.zeros((0,), dtype=np.int64)
            return empty_pts, empty_idx

        delta = (z_pc_final - z_img)
        delta0 = delta[keep0]
        med = float(np.median(delta0))
        mad = float(np.median(np.abs(delta0 - med)))
        sigma = 1.4826 * mad
        tau = max(float(depth_tau_min), float(depth_tau_k) * float(sigma))

        keep = keep0 & (np.abs(delta - med) < tau)

        if depth_tau_front is not None and float(depth_tau_front) > 0:
            keep = keep & (z_pc_final < (z_img + float(depth_tau_front)))

        if not np.any(keep):
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_idx = np.zeros((0,), dtype=np.int64)
            return empty_pts, empty_idx

        selected_indices = world_indices_final[keep].astype(np.int64)
        return world_pts[selected_indices].astype(np.float32), selected_indices

    selected_indices = world_indices_final[in_mask].astype(np.int64)
    return world_pts[selected_indices].astype(np.float32), selected_indices


# -------------------------
# DBSCAN helpers
# -------------------------

def dbscan_largest_cluster_centroid(
    pts: np.ndarray,
    eps: float,
    min_samples: int,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """
    Run DBSCAN on Nx3 points and return centroid of the largest cluster.
    Returns:
      centroid (3,) or None
      info dict: {num_points, num_clusters, largest_cluster_points, largest_cluster_ratio, noise_points}
    Notes:
      Uses open3d.cluster_dbscan if available.
      If open3d not available, returns centroid of all points (no clustering).
    """
    if pts is None or pts.size == 0:
        return None, {"num_points": 0, "num_clusters": 0, "largest_cluster_points": 0, "largest_cluster_ratio": 0.0, "noise_points": 0}

    pts = np.asarray(pts, dtype=np.float64)
    n = int(pts.shape[0])

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # labels: -1 = noise
        labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_samples), print_progress=False), dtype=np.int32)

        num_noise = int(np.sum(labels < 0))
        valid = labels >= 0
        if not np.any(valid):
            # all noise -> fallback to global centroid (still deterministic)
            c = pts.mean(axis=0)
            info = {
                "num_points": n,
                "num_clusters": 0,
                "largest_cluster_points": 0,
                "largest_cluster_ratio": 0.0,
                "noise_points": num_noise,
            }
            return c.astype(np.float32), info

        # find largest cluster id
        ids, counts = np.unique(labels[valid], return_counts=True)
        best_id = int(ids[int(np.argmax(counts))])
        best_cnt = int(np.max(counts))

        pts_best = pts[labels == best_id]
        c = pts_best.mean(axis=0)

        info = {
            "num_points": n,
            "num_clusters": int(len(ids)),
            "largest_cluster_points": best_cnt,
            "largest_cluster_ratio": float(best_cnt) / float(n),
            "noise_points": num_noise,
        }
        return c.astype(np.float32), info

    except Exception:
        # open3d not available or failed -> fallback
        c = pts.mean(axis=0)
        info = {
            "num_points": n,
            "num_clusters": 0,
            "largest_cluster_points": n,
            "largest_cluster_ratio": 1.0,
            "noise_points": 0,
        }
        return c.astype(np.float32), info


# -------------------------
# PLY saving helpers
# -------------------------

def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    synonyms = {
        "drawers": "drawer",
        "handles": "handle",
        "doors": "door",
        "cabinets": "cabinet",
    }
    for plural, singular in synonyms.items():
        if name == plural:
            return singular
    return name


def safe_filename(s: str) -> str:
    s = _normalize_name(s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unknown"


def save_points_as_ply(points_xyz: np.ndarray, out_path: Path, voxel_down: float = 0.0) -> None:
    if points_xyz is None or points_xyz.size == 0:
        return
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("Saving PLY requires open3d. Install: pip install open3d") from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))

    if voxel_down and voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(voxel_down))

    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False)


def save_xyz_as_ply(points_xyz: np.ndarray, out_path: Path) -> None:
    """Save Nx3 xyz as a point cloud ply (no colors)."""
    if points_xyz is None or points_xyz.size == 0:
        return
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError("Saving PLY requires open3d. Install: pip install open3d") from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False)
