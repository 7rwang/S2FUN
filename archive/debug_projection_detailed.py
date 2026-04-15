#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
import re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    import math
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).astype(np.float64)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

def read_traj_Twc(traj_path: Path):
    T_list = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no > 5:  # 只读前5个用于调试
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                _, ax, ay, az, tx, ty, tz = map(float, parts)
                R = rodrigues(np.array([ax, ay, az]))
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                T_list.append(T)
            except Exception:
                continue
    return T_list

def load_intrinsics_txt(p: Path):
    vals = p.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Intrinsics {p}: expect 6 values, got {len(vals)}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:])
    return w, h, fx, fy, cx, cy

def load_pcd_points(pcd_path: Path) -> np.ndarray:
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points, dtype=np.float32)
        return pts
    except Exception as e:
        print(f"Error loading {pcd_path}: {e}")
        return np.array([])

def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T

def project_points_to_image(world_pts: np.ndarray, T_cw: np.ndarray, w: int, h: int, fx: float, fy: float, cx: float, cy: float):
    print(f"Input points shape: {world_pts.shape}")
    print(f"Input points range: min={world_pts.min(axis=0)}, max={world_pts.max(axis=0)}")
    
    # Transform to camera coordinates
    pts_cam = apply_T(T_cw, world_pts)
    print(f"After camera transform range: min={pts_cam.min(axis=0)}, max={pts_cam.max(axis=0)}")
    
    z = pts_cam[:, 2]
    print(f"Z values: min={z.min()}, max={z.max()}, positive={np.sum(z > 0)}/{len(z)}")
    
    # Filter points behind camera and invalid
    valid = (z > 0) & np.isfinite(z) & np.isfinite(pts_cam).all(axis=1)
    print(f"Valid points after filtering: {np.sum(valid)}/{len(valid)}")
    
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])
    
    pts_v = pts_cam[valid]
    z_v = np.maximum(z[valid], 1e-6).astype(np.float32)
    
    # Project to image
    u_f = pts_v[:, 0] * fx / z_v + cx
    v_f = pts_v[:, 1] * fy / z_v + cy
    
    print(f"Projected u range: min={u_f.min()}, max={u_f.max()}")
    print(f"Projected v range: min={v_f.min()}, max={v_f.max()}")
    print(f"Image size: {w}x{h}")
    
    # Check finite
    ok = np.isfinite(u_f) & np.isfinite(v_f)
    print(f"Finite projected points: {np.sum(ok)}/{len(ok)}")
    
    if not np.any(ok):
        return np.array([]), np.array([]), np.array([])
        
    u_i = np.round(u_f[ok]).astype(np.int32)
    v_i = np.round(v_f[ok]).astype(np.int32)
    
    # Check bounds
    inside = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    print(f"Points inside image bounds: {np.sum(inside)}/{len(inside)}")
    
    u_final = u_i[inside]
    v_final = v_i[inside]
    
    return u_final, v_final, None

def main():
    scene_id = "435324"
    data_root = Path("/nas/qirui/scenefun3d/data")
    scene_root = data_root / scene_id
    
    # 找到序列目录
    seq_dirs = [p for p in scene_root.iterdir() if p.is_dir()]
    seq_dir = sorted(seq_dirs, key=lambda p: natural_key(p.name))[0]
    print(f"Using sequence: {seq_dir}")
    
    # 加载相机数据
    intr_dir = seq_dir / "hires_wide_intrinsics"
    traj_path = seq_dir / "hires_poses.traj"
    
    T_cw_list = read_traj_Twc(traj_path)
    intr_files = sorted([p for p in intr_dir.iterdir() if p.is_file()], key=lambda p: natural_key(p.name))
    
    print(f"Loaded {len(T_cw_list)} poses, {len(intr_files)} intrinsics")
    
    # 加载输出变换
    transform_path = seq_dir / f"{seq_dir.name}_transform.npy"
    T_out = np.load(str(transform_path)).astype(np.float64)
    T_inv = np.linalg.inv(T_out)
    
    # 加载预测点云
    scene_graph_path = Path("/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/435324/scene_graph.json")
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    
    pcd_path = Path(scene_graph["pcd_path"])
    predicted_points = load_pcd_points(pcd_path)
    
    # 加载点索引
    indices_path = scene_graph_path.with_name(scene_graph_path.stem + "_point_indices.json")
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
        point_indices_map = {node["id"]: node["point_indices"] for node in indices_data["node_point_indices"]}
    
    # 获取使用的点
    all_indices = set()
    for indices_list in point_indices_map.values():
        all_indices.update(indices_list)
    used_points = predicted_points[sorted(all_indices)]
    
    # 应用逆变换
    points_for_projection = apply_T(T_inv, used_points)
    
    print("\n=== Testing frame 0 projection ===")
    w, h, fx, fy, cx, cy = load_intrinsics_txt(intr_files[0])
    T_cw = T_cw_list[0]
    
    print(f"Camera intrinsics: {w}x{h}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"Camera pose T_cw:\n{T_cw}")
    
    u, v, _ = project_points_to_image(points_for_projection, T_cw, w, h, fx, fy, cx, cy)
    
    print(f"\nFinal projected points: {len(u)}")
    if len(u) > 0:
        print(f"u range: {u.min()} to {u.max()}")
        print(f"v range: {v.min()} to {v.max()}")
    
    # 测试GT点云投影
    print("\n=== Testing GT projection ===")
    gt_dir = Path("/nas/qirui/scenefun3d/ann_pcds/435324")
    ply_files = list(gt_dir.glob("*.ply"))
    if ply_files:
        gt_points = load_pcd_points(ply_files[0])
        u_gt, v_gt, _ = project_points_to_image(gt_points, T_cw, w, h, fx, fy, cx, cy)
        print(f"GT projected points: {len(u_gt)}")
        if len(u_gt) > 0:
            print(f"GT u range: {u_gt.min()} to {u_gt.max()}")
            print(f"GT v range: {v_gt.min()} to {v_gt.max()}")

if __name__ == "__main__":
    main()