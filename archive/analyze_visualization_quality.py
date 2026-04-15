#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_visualization_quality.py

分析3D mask质量问题的可视化诊断工具
帮助判断问题来源：2D mask质量 vs 预测点云质量
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def analyze_mask_quality(mask_path):
    """分析SAM mask的质量"""
    img = cv2.imread(str(mask_path))
    if img is None:
        return None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 统计mask属性
    total_pixels = gray.shape[0] * gray.shape[1]
    mask_pixels = np.sum(gray > 0)
    coverage_ratio = mask_pixels / total_pixels
    
    # 分析mask分布
    if mask_pixels > 0:
        coords = np.where(gray > 0)
        center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
        
        # 计算mask的紧凑性
        if len(coords[0]) > 1:
            y_spread = np.std(coords[0])
            x_spread = np.std(coords[1])
        else:
            y_spread = x_spread = 0
    else:
        center_y = center_x = y_spread = x_spread = 0
    
    # 统计不同颜色（物体类型）
    unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
    
    return {
        "coverage_ratio": coverage_ratio,
        "mask_pixels": mask_pixels,
        "center": (center_x, center_y),
        "spread": (x_spread, y_spread),
        "unique_colors": unique_colors,
        "compactness": mask_pixels / max(1, (x_spread + y_spread))
    }

def analyze_projection_quality(proj_path):
    """分析点云投影的质量"""
    img = cv2.imread(str(proj_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 统计投影属性
    total_pixels = gray.shape[0] * gray.shape[1]
    proj_pixels = np.sum(gray > 0)
    coverage_ratio = proj_pixels / total_pixels
    
    # 分析投影分布
    if proj_pixels > 0:
        coords = np.where(gray > 0)
        center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
        y_spread = np.std(coords[0]) if len(coords[0]) > 1 else 0
        x_spread = np.std(coords[1]) if len(coords[1]) > 1 else 0
        
        # 计算投影密度
        if len(coords[0]) > 100:
            # 采样计算局部密度
            sample_size = min(1000, len(coords[0]))
            sample_indices = np.random.choice(len(coords[0]), sample_size, replace=False)
            sample_coords = [(coords[0][i], coords[1][i]) for i in sample_indices]
            
            # 计算平均邻近距离
            from scipy.spatial.distance import pdist
            if len(sample_coords) > 1:
                distances = pdist(sample_coords)
                avg_distance = np.mean(distances)
                density_score = 1.0 / max(avg_distance, 1.0)
            else:
                density_score = 0.0
        else:
            density_score = 0.0
    else:
        center_y = center_x = y_spread = x_spread = density_score = 0
    
    return {
        "coverage_ratio": coverage_ratio,
        "proj_pixels": proj_pixels,
        "center": (center_x, center_y),
        "spread": (x_spread, y_spread), 
        "density_score": density_score
    }

def compute_overlap_metrics(mask_path, proj_path):
    """计算mask和投影的重叠度量"""
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    proj_img = cv2.imread(str(proj_path), cv2.IMREAD_GRAYSCALE)
    
    if mask_img is None or proj_img is None:
        return None
    
    # 二值化
    mask_binary = (mask_img > 0).astype(np.uint8)
    proj_binary = (proj_img > 0).astype(np.uint8)
    
    # 计算重叠
    intersection = np.sum(mask_binary & proj_binary)
    union = np.sum(mask_binary | proj_binary)
    mask_area = np.sum(mask_binary)
    proj_area = np.sum(proj_binary)
    
    # 各种重叠指标
    iou = intersection / max(union, 1)
    precision = intersection / max(proj_area, 1)  # 投影中有多少在mask内
    recall = intersection / max(mask_area, 1)     # mask中有多少被投影覆盖
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "intersection": intersection,
        "mask_area": mask_area,
        "proj_area": proj_area
    }

def diagnose_quality_issues(mask_stats, pred_stats, gt_stats, overlap_pred, overlap_gt):
    """诊断质量问题的根因"""
    issues = []
    recommendations = []
    
    # 1. 检查2D mask质量
    if mask_stats and mask_stats["coverage_ratio"] < 0.001:
        issues.append("❌ SAM mask几乎为空")
        recommendations.append("检查SAM分割质量或mask文件")
    elif mask_stats and mask_stats["coverage_ratio"] > 0.5:
        issues.append("⚠️ SAM mask覆盖过大，可能过分割")
        recommendations.append("调整SAM参数或后处理")
    
    # 2. 检查预测点云投影
    if pred_stats and pred_stats["coverage_ratio"] < 0.01:
        issues.append("❌ 预测点云投影稀疏")
        recommendations.append("检查affordance点云数量或投影逻辑")
    elif pred_stats and pred_stats["coverage_ratio"] > 0.3:
        issues.append("⚠️ 预测点云投影过密，可能过'胖'")
        recommendations.append("检查点云质量或调整过滤参数")
    
    # 3. 检查重叠度
    if overlap_pred and overlap_pred["iou"] < 0.1:
        issues.append("❌ 预测投影与mask重叠度很低")
        if overlap_pred["precision"] < 0.3:
            recommendations.append("预测点云位置偏移，检查坐标变换")
        if overlap_pred["recall"] < 0.3:
            recommendations.append("预测点云覆盖不足，检查点云完整性")
    
    # 4. 对比GT
    if overlap_gt and overlap_pred:
        if overlap_gt["iou"] > overlap_pred["iou"] * 1.5:
            issues.append("⚠️ GT投影明显好于预测，说明预测质量有问题")
            recommendations.append("重点检查预测算法和点云质量")
        elif overlap_pred["iou"] > overlap_gt["iou"] * 1.5:
            issues.append("⚠️ 预测投影好于GT，可能GT标注有问题")
            recommendations.append("检查GT数据质量")
    
    return issues, recommendations

def analyze_single_frame(frame_dir):
    """分析单个帧的质量"""
    mask_path = frame_dir / "sam_masks.png"
    pred_path = frame_dir / "predicted_projection.png"
    gt_path = frame_dir / "groundtruth_projection.png"
    
    # 分析各个组件
    mask_stats = analyze_mask_quality(mask_path) if mask_path.exists() else None
    pred_stats = analyze_projection_quality(pred_path) if pred_path.exists() else None
    gt_stats = analyze_projection_quality(gt_path) if gt_path.exists() else None
    
    # 计算重叠
    overlap_pred = compute_overlap_metrics(mask_path, pred_path) if mask_path.exists() and pred_path.exists() else None
    overlap_gt = compute_overlap_metrics(mask_path, gt_path) if mask_path.exists() and gt_path.exists() else None
    
    # 诊断问题
    issues, recommendations = diagnose_quality_issues(mask_stats, pred_stats, gt_stats, overlap_pred, overlap_gt)
    
    return {
        "mask_stats": mask_stats,
        "pred_stats": pred_stats,
        "gt_stats": gt_stats,
        "overlap_pred": overlap_pred,
        "overlap_gt": overlap_gt,
        "issues": issues,
        "recommendations": recommendations
    }

def print_analysis_report(scene_id, frame_id, analysis):
    """打印分析报告"""
    print(f"\n{'='*60}")
    print(f"场景 {scene_id} - 帧 {frame_id} 质量分析")
    print(f"{'='*60}")
    
    # SAM Mask统计
    if analysis["mask_stats"]:
        stats = analysis["mask_stats"]
        print(f"📋 SAM Mask:")
        print(f"  覆盖率: {stats['coverage_ratio']*100:.2f}%")
        print(f"  像素数: {stats['mask_pixels']}")
        print(f"  中心点: ({stats['center'][0]:.0f}, {stats['center'][1]:.0f})")
        print(f"  分布范围: {stats['spread'][0]:.1f} x {stats['spread'][1]:.1f}")
        print(f"  颜色数: {stats['unique_colors']}")
    
    # 预测投影统计
    if analysis["pred_stats"]:
        stats = analysis["pred_stats"]
        print(f"🎯 预测投影:")
        print(f"  覆盖率: {stats['coverage_ratio']*100:.2f}%")
        print(f"  像素数: {stats['proj_pixels']}")
        print(f"  中心点: ({stats['center'][0]:.0f}, {stats['center'][1]:.0f})")
        print(f"  分布范围: {stats['spread'][0]:.1f} x {stats['spread'][1]:.1f}")
        print(f"  密度分数: {stats['density_score']:.3f}")
    
    # GT投影统计
    if analysis["gt_stats"]:
        stats = analysis["gt_stats"]
        print(f"✅ GT投影:")
        print(f"  覆盖率: {stats['coverage_ratio']*100:.2f}%")
        print(f"  像素数: {stats['proj_pixels']}")
    
    # 重叠分析
    if analysis["overlap_pred"]:
        overlap = analysis["overlap_pred"]
        print(f"🎭 预测-Mask重叠:")
        print(f"  IoU: {overlap['iou']:.3f}")
        print(f"  精确度: {overlap['precision']:.3f} (投影中{overlap['precision']*100:.1f}%在mask内)")
        print(f"  召回率: {overlap['recall']:.3f} (mask中{overlap['recall']*100:.1f}%被覆盖)")
    
    if analysis["overlap_gt"]:
        overlap = analysis["overlap_gt"]
        print(f"🎯 GT-Mask重叠:")
        print(f"  IoU: {overlap['iou']:.3f}")
        print(f"  精确度: {overlap['precision']:.3f}")
        print(f"  召回率: {overlap['recall']:.3f}")
    
    # 问题诊断
    if analysis["issues"]:
        print(f"\n⚠️ 发现的问题:")
        for issue in analysis["issues"]:
            print(f"  {issue}")
    
    if analysis["recommendations"]:
        print(f"\n💡 建议:")
        for rec in analysis["recommendations"]:
            print(f"  • {rec}")

def main():
    parser = argparse.ArgumentParser(description="分析3D mask可视化质量")
    parser.add_argument("--scene_id", type=str, required=True, help="场景ID")
    parser.add_argument("--output_dir", type=str, 
                       default="/nas/qirui/sam3/S2FUN/visualization_output_30scenes_final",
                       help="可视化输出目录")
    parser.add_argument("--frame_ids", type=str, default="0,10,20", 
                       help="要分析的帧ID，用逗号分隔")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    scene_dir = output_dir / args.scene_id
    
    if not scene_dir.exists():
        print(f"❌ 场景目录不存在: {scene_dir}")
        return
    
    frame_ids = [int(x.strip()) for x in args.frame_ids.split(",")]
    
    print(f"🔍 分析场景 {args.scene_id} 的可视化质量")
    print(f"📁 目录: {scene_dir}")
    
    for frame_id in frame_ids:
        frame_dir = scene_dir / f"frame_{frame_id:04d}"
        if frame_dir.exists():
            analysis = analyze_single_frame(frame_dir)
            print_analysis_report(args.scene_id, frame_id, analysis)
        else:
            print(f"\n❌ 帧目录不存在: {frame_dir}")

if __name__ == "__main__":
    main()