#!/usr/bin/env python3

import numpy as np
import cv2
from pathlib import Path
import json

def load_pcd_points(pcd_path: Path) -> np.ndarray:
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points, dtype=np.float32)
        return pts
    except Exception as e:
        print(f"Error loading {pcd_path}: {e}")
        return np.array([])

def test_simple_projection():
    # 使用已知有点投影到图像内的测试数据
    print("=== Simple Test ===")
    
    # 创建一些简单的测试点（在相机前方）
    test_points = np.array([
        [0, 0, 5],      # 直接在相机前方5米
        [1, 0, 5],      # 右侧1米
        [-1, 0, 5],     # 左侧1米
        [0, 1, 5],      # 上方1米
        [0, -1, 5],     # 下方1米
    ], dtype=np.float32)
    
    # 相机内参 (实际值)
    w, h = 1920, 1440
    fx, fy = 1592.47, 1592.47
    cx, cy = 955.752, 740.417
    
    # 单位变换矩阵 (相机在原点，朝向+Z)
    T_cw = np.eye(4, dtype=np.float64)
    
    # 投影
    pts_cam = test_points  # 已经在相机坐标系
    z = pts_cam[:, 2]
    print(f"Z values: {z}")
    
    u = pts_cam[:, 0] * fx / z + cx
    v = pts_cam[:, 1] * fy / z + cy
    
    print(f"Projected u: {u}")
    print(f"Projected v: {v}")
    
    # 创建图像并绘制点
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, (ui, vi) in enumerate(zip(u, v)):
        ui, vi = int(ui), int(vi)
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(image, (ui, vi), 10, (0, 255, 255), -1)  # 青色圆点
            print(f"Point {i}: ({ui}, {vi}) - VISIBLE")
        else:
            print(f"Point {i}: ({ui}, {vi}) - OUTSIDE")
    
    cv2.imwrite('/nas/qirui/sam3/S2FUN/test_simple.png', image)
    print("Test image saved to test_simple.png")

def test_actual_data():
    print("\n=== Testing Actual Data ===")
    
    # 加载真实场景数据
    scene_graph_path = Path("/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/435324/scene_graph.json")
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    
    # 加载点云
    pcd_path = Path(scene_graph["pcd_path"]) 
    predicted_points = load_pcd_points(pcd_path)
    
    # 加载点索引
    indices_path = scene_graph_path.with_name(scene_graph_path.stem + "_point_indices.json")
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
        point_indices_map = {node["id"]: node["point_indices"] for node in indices_data["node_point_indices"]}
    
    # 获取第一个节点的点
    first_node_id = list(point_indices_map.keys())[0]
    first_node_indices = point_indices_map[first_node_id]
    first_node_points = predicted_points[first_node_indices[:100]]  # 只取前100个点
    
    print(f"First node points shape: {first_node_points.shape}")
    print(f"First node points range: min={first_node_points.min(axis=0)}, max={first_node_points.max(axis=0)}")
    
    # 单位变换测试
    T_cw = np.eye(4, dtype=np.float64)
    w, h = 1920, 1440
    fx, fy = 1592.47, 1592.47
    cx, cy = 955.752, 740.417
    
    pts_cam = first_node_points
    z = pts_cam[:, 2]
    valid = z > 0
    
    print(f"Points with positive Z: {np.sum(valid)}/{len(valid)}")
    
    if np.sum(valid) > 0:
        pts_v = pts_cam[valid]
        z_v = pts_v[:, 2]
        u = pts_v[:, 0] * fx / z_v + cx
        v = pts_v[:, 1] * fy / z_v + cy
        
        print(f"u range: {u.min():.1f} to {u.max():.1f}")
        print(f"v range: {v.min():.1f} to {v.max():.1f}")
        
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        print(f"Points inside image: {np.sum(inside)}/{len(inside)}")

if __name__ == "__main__":
    test_simple_projection()
    test_actual_data()