#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

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
    # 测试scene 435324
    scene_graph_path = Path("/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/435324/scene_graph.json")
    
    # 加载scene graph
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    
    # 加载点云
    pcd_path = Path(scene_graph["pcd_path"])
    print(f"Loading point cloud: {pcd_path}")
    predicted_points = load_pcd_points(pcd_path)
    print(f"Original predicted points shape: {predicted_points.shape}")
    print(f"Original predicted points range: min={predicted_points.min(axis=0)}, max={predicted_points.max(axis=0)}")
    
    # 加载并检查点索引文件
    indices_path = scene_graph_path.with_name(scene_graph_path.stem + "_point_indices.json")
    if indices_path.exists():
        with open(indices_path, 'r') as f:
            indices_data = json.load(f)
            point_indices_map = {
                node["id"]: node["point_indices"] 
                for node in indices_data["node_point_indices"]
            }
        
        # 获取实际被使用的点
        all_indices = set()
        for indices_list in point_indices_map.values():
            all_indices.update(indices_list)
        
        if all_indices:
            used_points = predicted_points[sorted(all_indices)]
            print(f"Used predicted points shape: {used_points.shape}")
            print(f"Used predicted points range: min={used_points.min(axis=0)}, max={used_points.max(axis=0)}")
        else:
            print("No point indices found!")
    else:
        print("Point indices file not found!")
        used_points = predicted_points
    
    # 检查输出变换
    transform_path = Path("/nas/qirui/scenefun3d/data/435324/42899216/42899216_transform.npy")
    if transform_path.exists():
        T_out = np.load(str(transform_path)).astype(np.float64)
        print(f"Output transform:\n{T_out}")
        
        # 逆变换测试
        T_inv = np.linalg.inv(T_out)
        print(f"Inverse transform:\n{T_inv}")
        
        # 变换点云
        if 'used_points' in locals():
            points_transformed = apply_T(T_inv, used_points)
            print(f"After inverse transform: min={points_transformed.min(axis=0)}, max={points_transformed.max(axis=0)}")
            
            # 检查是否有无效值
            has_inf = np.isinf(points_transformed).any()
            has_nan = np.isnan(points_transformed).any()
            print(f"Has inf: {has_inf}, Has nan: {has_nan}")
        
    # 加载GT点云进行对比
    gt_dir = Path("/nas/qirui/scenefun3d/ann_pcds/435324")
    if gt_dir.exists():
        ply_files = list(gt_dir.glob("*.ply"))
        all_gt_points = []
        for ply_path in ply_files[:3]:  # 只加载前3个
            gt_points = load_pcd_points(ply_path)
            if gt_points.size > 0:
                all_gt_points.append(gt_points)
        
        if all_gt_points:
            gt_combined = np.vstack(all_gt_points)
            print(f"GT points shape: {gt_combined.shape}")
            print(f"GT points range: min={gt_combined.min(axis=0)}, max={gt_combined.max(axis=0)}")

if __name__ == "__main__":
    main()