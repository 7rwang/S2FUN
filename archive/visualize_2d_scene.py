#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_2d_scene.py

2D Visualization script for SAM3 + Point Cloud projections + Ground Truth
Creates frame-by-frame visualizations with three components:
1. SAM3 masks
2. Predicted point cloud projections 
3. Ground truth point cloud projections

Usage:
python visualize_2d_scene.py \
  --scene_id 435324 \
  --masks_root /nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv \
  --data_root /nas/qirui/scenefun3d/data \
  --scene_graph_json /path/to/scene_graph.json \
  --gt_pcds_root /nas/qirui/scenefun3d/ann_pcds \
  --output_dir /path/to/output
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# ─────────────────────────────────────────────
# Utility functions (extracted from build_scene_graph.py)
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


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def load_mask(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


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


# ─────────────────────────────────────────────
# Scene discovery functions  
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


# ─────────────────────────────────────────────
# Projection functions
# ─────────────────────────────────────────────

def project_points_to_image(
    world_pts: np.ndarray,
    T_cw: np.ndarray,
    w: int, h: int,
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D world points to 2D image coordinates.
    
    Returns:
        u: x coordinates in image
        v: y coordinates in image  
        valid_mask: boolean mask for points that project inside image bounds
    """
    # Transform to camera coordinates
    pts_cam = apply_T(T_cw, world_pts)
    z = pts_cam[:, 2]
    
    # Filter points behind camera and invalid
    valid = (z > 0) & np.isfinite(z) & np.isfinite(pts_cam).all(axis=1)
    
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])
    
    pts_v = pts_cam[valid]
    z_v = np.maximum(z[valid], 1e-6).astype(np.float32)
    
    # Project to image
    u_f = pts_v[:, 0] * fx / z_v + cx
    v_f = pts_v[:, 1] * fy / z_v + cy
    
    # Check finite
    ok = np.isfinite(u_f) & np.isfinite(v_f)
    u_i = np.round(u_f[ok]).astype(np.int32)
    v_i = np.round(v_f[ok]).astype(np.int32)
    
    # Check bounds
    inside = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)
    u_final = u_i[inside]
    v_final = v_i[inside]
    
    # Create full valid mask
    full_valid_mask = np.zeros(len(world_pts), dtype=bool)
    valid_indices = np.where(valid)[0]
    ok_indices = valid_indices[ok]
    inside_indices = ok_indices[inside]
    full_valid_mask[inside_indices] = True
    
    return u_final, v_final, full_valid_mask


# ─────────────────────────────────────────────
# Scene Graph Loader
# ─────────────────────────────────────────────

class SceneGraphLoader:
    def __init__(self, scene_graph_path: Path, scene_root: Path):
        self.scene_graph_path = scene_graph_path
        self.scene_root = scene_root
        self.scene_graph = None
        self.predicted_points = None
        self.point_indices_map = None
        
    def load(self):
        """Load scene graph and extract predicted point clouds."""
        print(f"Loading scene graph: {self.scene_graph_path}")
        
        # Load scene graph JSON
        with open(self.scene_graph_path, 'r') as f:
            self.scene_graph = json.load(f)
            
        # Load point indices JSON (companion file)
        indices_path = self.scene_graph_path.with_name(
            self.scene_graph_path.stem + "_point_indices.json"
        )
        if indices_path.exists():
            with open(indices_path, 'r') as f:
                indices_data = json.load(f)
                self.point_indices_map = {
                    node["id"]: node["point_indices"] 
                    for node in indices_data["node_point_indices"]
                }
        
        # Load original point cloud
        pcd_path = Path(self.scene_graph["pcd_path"])
        print(f"Loading predicted point cloud: {pcd_path}")
        self.predicted_points = load_pcd_points(pcd_path)
        
        print(f"Loaded {len(self.scene_graph['nodes'])} nodes with {len(self.predicted_points)} points")
        
    def get_node_points(self, node_id: int) -> np.ndarray:
        """Get 3D points for a specific node."""
        if self.point_indices_map is None or node_id not in self.point_indices_map:
            return np.empty((0, 3))
        
        indices = self.point_indices_map[node_id]
        return self.predicted_points[indices]
        
    def get_all_predicted_points(self) -> np.ndarray:
        """Get all predicted points as single array."""
        # Use original point cloud directly for better visualization coverage
        return self.predicted_points
    
    def get_affordance_points(self) -> np.ndarray:
        """Get only affordance points for visualization."""
        if self.point_indices_map is None or self.scene_graph is None:
            return np.empty((0, 3))
        
        # Find all affordance nodes
        affordance_indices = set()
        for node in self.scene_graph.get("nodes", []):
            if node.get("type") == "affordance":
                node_id = node.get("id")
                if node_id in self.point_indices_map:
                    affordance_indices.update(self.point_indices_map[node_id])
        
        if affordance_indices:
            return self.predicted_points[sorted(affordance_indices)]
        return np.empty((0, 3))


# ─────────────────────────────────────────────
# Ground Truth Loader  
# ─────────────────────────────────────────────

class GroundTruthLoader:
    def __init__(self, gt_pcds_root: Path, scene_id: str):
        self.gt_pcds_root = gt_pcds_root
        self.scene_id = scene_id
        self.gt_points = None
        
    def load(self):
        """Load all ground truth point clouds for the scene."""
        gt_scene_dir = self.gt_pcds_root / self.scene_id
        if not gt_scene_dir.exists():
            print(f"Warning: Ground truth directory not found: {gt_scene_dir}")
            self.gt_points = np.empty((0, 3))
            return
            
        ply_files = list(gt_scene_dir.glob("*.ply"))
        if not ply_files:
            print(f"Warning: No PLY files found in: {gt_scene_dir}")
            self.gt_points = np.empty((0, 3))
            return
            
        print(f"Loading {len(ply_files)} ground truth PLY files from: {gt_scene_dir}")
        
        all_gt_points = []
        for ply_path in ply_files:
            try:
                points = load_pcd_points(ply_path)
                all_gt_points.append(points)
            except Exception as e:
                print(f"Warning: Failed to load {ply_path}: {e}")
                
        if all_gt_points:
            self.gt_points = np.vstack(all_gt_points).astype(np.float32)
        else:
            self.gt_points = np.empty((0, 3))
            
        print(f"Loaded {len(self.gt_points)} ground truth points")


# ─────────────────────────────────────────────
# Visualization Generator
# ─────────────────────────────────────────────

class VisualizationGenerator:
    def __init__(self, output_dir: Path, scene_id: str):
        self.output_dir = output_dir
        self.scene_id = scene_id
        self.scene_output_dir = output_dir / scene_id
        self.scene_output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sam_mask_visualization(self, mask_paths: List[Path], frame_idx: int, w: int, h: int) -> np.ndarray:
        """Create combined SAM mask visualization."""
        # Create composite mask image
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Different colors for different mask types
        object_color = (0, 255, 0)    # Green for objects
        affordance_color = (255, 0, 0)  # Red for affordances
        
        for mask_path in mask_paths:
            try:
                mask = load_mask(mask_path)
                
                # Determine mask type from filename
                if mask_path.stem.startswith("CTX"):
                    color = object_color
                elif mask_path.stem.startswith("INT"):
                    color = affordance_color
                else:
                    color = object_color  # Default to object
                
                # Apply color where mask is active
                for c in range(3):
                    composite[mask > 0, c] = color[c]
                    
            except Exception as e:
                print(f"Warning: Failed to process mask {mask_path}: {e}")
                
        return composite
    
    def create_point_projection_visualization(self, points: np.ndarray, T_cw: np.ndarray,
                                            w: int, h: int, fx: float, fy: float, cx: float, cy: float,
                                            color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Create point cloud projection visualization."""
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if points.size == 0:
            return image
            
        # Project points to image
        u, v, valid_mask = project_points_to_image(points, T_cw, w, h, fx, fy, cx, cy)
        
        # Draw points
        if len(u) > 0:
            for ui, vi in zip(u, v):
                cv2.circle(image, (int(ui), int(vi)), 1, color, -1)
                
        return image
        
    def save_frame_visualizations(self, frame_idx: int, sam_vis: np.ndarray, 
                                pred_vis: np.ndarray, gt_vis: np.ndarray):
        """Save the three visualizations for a frame."""
        frame_dir = self.scene_output_dir / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(frame_dir / "sam_masks.png"), sam_vis)
        cv2.imwrite(str(frame_dir / "predicted_projection.png"), pred_vis)
        cv2.imwrite(str(frame_dir / "groundtruth_projection.png"), gt_vis)


# ─────────────────────────────────────────────
# Frame Processor
# ─────────────────────────────────────────────

class FrameProcessor:
    def __init__(self, scene_id: str, masks_root: Path, data_root: Path,
                 scene_graph_loader: SceneGraphLoader, gt_loader: GroundTruthLoader,
                 vis_generator: VisualizationGenerator):
        self.scene_id = scene_id
        self.masks_root = masks_root
        self.data_root = data_root
        self.scene_graph_loader = scene_graph_loader
        self.gt_loader = gt_loader
        self.vis_generator = vis_generator
        
        # Load scene data
        self.scene_root = data_root / scene_id
        self.seq_dir = find_first_sequence_dir(self.scene_root)
        self.intr_dir = find_intr_dir(self.seq_dir)
        self.traj_path = find_traj_path(self.seq_dir)
        
        print(f"Loading camera data for scene {scene_id}")
        self.T_cw_list = read_traj_Twc(self.traj_path)
        
        self.intr_files = sorted([p for p in self.intr_dir.iterdir() if p.is_file()],
                                key=lambda p: natural_key(p.name))
        
        # Load output transform if available
        self.T_out = np.eye(4, dtype=np.float64)
        transform_path = find_transform_npy(self.seq_dir)
        if transform_path and transform_path.exists():
            print(f"Loading transform: {transform_path}")
            self.T_out = np.load(str(transform_path)).astype(np.float64)
            validate_transform(self.T_out, "Output transform")
        
        print(f"Loaded {len(self.T_cw_list)} poses, {len(self.intr_files)} intrinsics")
        
    def process_all_frames(self, frame_stride: int = 1):
        """Process all frames and generate visualizations."""
        # Get frame list
        scene_masks_root = self.masks_root / self.scene_id / "masks"
        frame_ids_all = list_frame_dirs(scene_masks_root)
        frame_ids = frame_ids_all[::max(1, frame_stride)]
        
        print(f"Processing {len(frame_ids)} frames (stride {frame_stride})")
        
        # Get predicted and ground truth points
        # Note: Only use affordance points for prediction visualization
        predicted_points = self.scene_graph_loader.get_affordance_points()
        print(f"Using {len(predicted_points)} affordance points for visualization")
            
        gt_points = self.gt_loader.gt_points
        
        for i, frame_idx in enumerate(frame_ids):
            print(f"Processing frame {frame_idx} ({i+1}/{len(frame_ids)})")
            
            # Load camera parameters for this frame
            w, h, fx, fy, cx, cy = load_intrinsics_txt(self.intr_files[frame_idx])
            T_cw = self.T_cw_list[frame_idx]
            
            # Load SAM masks for this frame
            frame_dir = get_frame_dir(scene_masks_root, frame_idx)
            mask_paths = sorted(
                [p for p in frame_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")],
                key=lambda p: natural_key(p.name)
            )
            
            # Generate visualizations
            sam_vis = self.vis_generator.create_sam_mask_visualization(mask_paths, frame_idx, w, h)
            
            pred_vis = self.vis_generator.create_point_projection_visualization(
                predicted_points, T_cw, w, h, fx, fy, cx, cy, color=(0, 255, 255)  # Cyan for predicted
            )
            
            gt_vis = self.vis_generator.create_point_projection_visualization(
                gt_points, T_cw, w, h, fx, fy, cx, cy, color=(255, 255, 0)  # Yellow for GT
            )
            
            # Save results
            self.vis_generator.save_frame_visualizations(frame_idx, sam_vis, pred_vis, gt_vis)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D visualizations for SAM + predicted + ground truth point clouds"
    )
    parser.add_argument("--scene_id", type=str, required=True,
                       help="Scene ID to process")
    parser.add_argument("--masks_root", type=str, required=True,
                       help="Root directory containing SAM masks")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing scene data")
    parser.add_argument("--scene_graph_json", type=str, required=True,
                       help="Path to scene graph JSON file")
    parser.add_argument("--gt_pcds_root", type=str, required=True,
                       help="Root directory containing ground truth point clouds")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for visualizations")
    parser.add_argument("--frame_stride", type=int, default=1,
                       help="Process every Nth frame (default: 1)")
    
    args = parser.parse_args()
    
    # Convert paths
    scene_id = args.scene_id
    masks_root = Path(args.masks_root)
    data_root = Path(args.data_root)
    scene_graph_path = Path(args.scene_graph_json)
    gt_pcds_root = Path(args.gt_pcds_root)
    output_dir = Path(args.output_dir)
    
    print(f"=== 2D Visualization for Scene {scene_id} ===")
    
    # Initialize components
    scene_graph_loader = SceneGraphLoader(scene_graph_path, data_root / scene_id)
    gt_loader = GroundTruthLoader(gt_pcds_root, scene_id)
    vis_generator = VisualizationGenerator(output_dir, scene_id)
    
    # Load data
    scene_graph_loader.load()
    gt_loader.load()
    
    # Process frames
    frame_processor = FrameProcessor(
        scene_id, masks_root, data_root,
        scene_graph_loader, gt_loader, vis_generator
    )
    
    frame_processor.process_all_frames(args.frame_stride)
    
    print(f"=== Completed! Results saved to: {output_dir / scene_id} ===")


if __name__ == "__main__":
    main()