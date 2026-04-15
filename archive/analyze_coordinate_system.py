#!/usr/bin/env python3

import numpy as np
import json
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
            if line_no > 3:  # 只读前3个用于调试
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

def main():
    scene_id = "435324"
    print(f"=== 分析场景 {scene_id} 的坐标系问题 ===")
    
    # 1. 加载原始点云和变换
    scene_graph_path = Path("/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/435324/scene_graph.json")
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    
    # 加载原始点云
    pcd_path = Path(scene_graph["pcd_path"])
    print(f"原始点云路径: {pcd_path}")
    original_points = load_pcd_points(pcd_path)
    print(f"原始点云范围: min={original_points.min(axis=0)}, max={original_points.max(axis=0)}")
    
    # 2. 加载输出变换
    data_root = Path("/nas/qirui/scenefun3d/data")
    scene_root = data_root / scene_id
    seq_dirs = [p for p in scene_root.iterdir() if p.is_dir()]
    seq_dir = sorted(seq_dirs, key=lambda p: natural_key(p.name))[0]
    transform_path = seq_dir / f"{seq_dir.name}_transform.npy"
    
    if transform_path.exists():
        T_out = np.load(str(transform_path)).astype(np.float64)
        print(f"输出变换矩阵:\n{T_out}")
    else:
        T_out = np.eye(4, dtype=np.float64)
        print("没有找到变换矩阵，使用单位矩阵")
    
    # 3. 加载相机轨迹
    traj_path = seq_dir / "hires_poses.traj"
    T_cw_list = read_traj_Twc(traj_path)
    print(f"加载了 {len(T_cw_list)} 个相机姿态")
    
    # 4. 测试不同的变换策略
    print("\n=== 测试不同的变换策略 ===")
    
    # 取一小部分点进行测试
    test_indices = np.random.choice(len(original_points), min(1000, len(original_points)), replace=False)
    test_points = original_points[test_indices]
    
    strategies = [
        ("原始点云", test_points),
        ("应用输出变换", apply_T(T_out, test_points)),
        ("应用逆输出变换", apply_T(np.linalg.inv(T_out), test_points)),
    ]
    
    # 使用第一个相机姿态进行测试
    if T_cw_list:
        T_cw = T_cw_list[0]
        print(f"使用相机姿态:\n{T_cw}")
        
        for strategy_name, points in strategies:
            print(f"\n--- 策略: {strategy_name} ---")
            print(f"点云范围: min={points.min(axis=0)}, max={points.max(axis=0)}")
            
            # 变换到相机坐标系
            pts_cam = apply_T(T_cw, points)
            z = pts_cam[:, 2]
            positive_z = np.sum(z > 0)
            print(f"相机坐标系Z值: min={z.min():.3f}, max={z.max():.3f}")
            print(f"正Z值(可见)点数: {positive_z}/{len(z)} ({positive_z/len(z)*100:.1f}%)")
            
            if positive_z > 0:
                # 计算投影
                valid = z > 0
                pts_v = pts_cam[valid]
                z_v = pts_v[:, 2]
                
                fx, fy = 1592.47, 1592.47
                cx, cy = 955.752, 740.417
                w, h = 1920, 1440
                
                u = pts_v[:, 0] * fx / z_v + cx
                v = pts_v[:, 1] * fy / z_v + cy
                
                inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                print(f"投影到图像内的点数: {np.sum(inside)}/{len(inside)} ({np.sum(inside)/len(inside)*100:.1f}%)")
                
                if np.sum(inside) > 0:
                    print(f"图像内点的u范围: {u[inside].min():.1f} ~ {u[inside].max():.1f}")
                    print(f"图像内点的v范围: {v[inside].min():.1f} ~ {v[inside].max():.1f}")
    
    # 5. 查看scene graph中节点的位置信息
    print(f"\n=== Scene Graph 节点信息 ===")
    print(f"节点数量: {len(scene_graph['nodes'])}")
    
    if scene_graph['nodes']:
        positions = [node['position'] for node in scene_graph['nodes'][:5]]  # 前5个
        positions = np.array(positions)
        print(f"前5个节点位置:")
        for i, pos in enumerate(positions):
            print(f"  节点{i}: {pos}")
        
        print(f"节点位置范围: min={positions.min(axis=0)}, max={positions.max(axis=0)}")
        
        # 测试这些位置的投影
        if T_cw_list:
            T_cw = T_cw_list[0]
            pts_cam = apply_T(T_cw, positions)
            z = pts_cam[:, 2]
            print(f"节点位置在相机坐标系的Z值: {z}")

if __name__ == "__main__":
    main()