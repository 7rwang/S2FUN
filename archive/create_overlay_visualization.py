#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_overlay_visualization.py

创建三种投影结果的叠加可视化，用于质量分析
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def load_and_process_mask(mask_path):
    """加载并处理SAM mask"""
    img = cv2.imread(str(mask_path))
    if img is None:
        return None, None
    
    # SAM mask中的颜色: BGR(0, 255, 0)绿色=object, BGR(255, 0, 0)红色=affordance
    green_mask = (img[:, :, 1] == 255) & (img[:, :, 0] == 0) & (img[:, :, 2] == 0)  # BGR绿色objects
    red_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)    # BGR红色affordances
    
    return green_mask.astype(np.uint8) * 255, red_mask.astype(np.uint8) * 255

def load_projection(proj_path):
    """加载投影图像并转为二值mask"""
    img = cv2.imread(str(proj_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 0).astype(np.uint8) * 255

def add_legend_to_image(img):
    """在图像左上角添加简洁的legend"""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Legend box尺寸
    legend_width = 180
    legend_height = 80
    
    # 创建半透明背景
    legend_bg = img[0:legend_height, 0:legend_width].copy()
    legend_bg = legend_bg * 0.3  # 让背景变暗
    
    # 添加legend项目
    y_start = 15
    line_height = 15
    
    # Affordance (红色背景)
    cv2.rectangle(legend_bg, (5, y_start-8), (15, y_start+2), [0, 0, 80], -1)
    cv2.putText(legend_bg, "Affordance Area", (20, y_start), font, font_scale, (255, 255, 255), thickness)
    
    # 预测点云 (青色)
    y_start += line_height
    cv2.rectangle(legend_bg, (5, y_start-8), (15, y_start+2), [255, 255, 0], -1)
    cv2.putText(legend_bg, "Predicted Points", (20, y_start), font, font_scale, (255, 255, 255), thickness)
    
    # GT点云 (黄色)
    y_start += line_height
    cv2.rectangle(legend_bg, (5, y_start-8), (15, y_start+2), [0, 255, 255], -1)
    cv2.putText(legend_bg, "GT Points", (20, y_start), font, font_scale, (255, 255, 255), thickness)
    
    # 重叠区域 (白色)
    y_start += line_height
    cv2.rectangle(legend_bg, (5, y_start-8), (15, y_start+2), [255, 255, 255], -1)
    cv2.putText(legend_bg, "Overlap", (20, y_start), font, font_scale, (255, 255, 255), thickness)
    
    # 将legend贴回原图
    img[0:legend_height, 0:legend_width] = legend_bg
    return img

def create_overlay_visualization(mask_path, pred_path, gt_path, output_path):
    """创建叠加可视化"""
    # 读取原始图像
    mask_img = cv2.imread(str(mask_path))
    if mask_img is None:
        print(f"无法加载mask: {mask_path}")
        return False
    
    h, w = mask_img.shape[:2]
    
    # 创建输出图像
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. 加载SAM masks (作为背景)
    green_mask, red_mask = load_and_process_mask(mask_path)
    if green_mask is not None:
        # 绿色mask (objects) - 半透明绿色背景
        overlay[green_mask > 0] = [0, 80, 0]  # 深绿色背景
        # 红色mask (affordances) - 半透明红色背景  
        overlay[red_mask > 0] = [0, 0, 80]   # 深红色背景
    
    # 2. 加载预测投影 (青色点)
    pred_mask = load_projection(pred_path)
    if pred_mask is not None:
        overlay[pred_mask > 0] = [255, 255, 0]  # 青色点
    
    # 3. 加载GT投影 (黄色点)  
    gt_mask = load_projection(gt_path)
    if gt_mask is not None:
        # 如果GT和预测重叠，用白色表示
        overlap = (pred_mask > 0) & (gt_mask > 0) if pred_mask is not None else (gt_mask > 0)
        gt_only = (gt_mask > 0) & (pred_mask == 0) if pred_mask is not None else (gt_mask > 0)
        
        overlay[overlap] = [255, 255, 255]  # 白色 = 完美重叠
        overlay[gt_only] = [0, 255, 255]    # 黄色 = GT独有
    
    # 添加legend到图像
    overlay_with_legend = add_legend_to_image(overlay)
    
    # 保存结果
    cv2.imwrite(str(output_path), overlay_with_legend)
    return True

def analyze_overlap_statistics(mask_path, pred_path, gt_path):
    """分析重叠统计信息"""
    # 加载所有mask
    green_mask, red_mask = load_and_process_mask(mask_path)
    pred_mask = load_projection(pred_path)
    gt_mask = load_projection(gt_path)
    
    if green_mask is None or pred_mask is None or gt_mask is None:
        return None
    
    # 计算各种区域
    affordance_area = np.sum(red_mask > 0)
    object_area = np.sum(green_mask > 0)
    pred_area = np.sum(pred_mask > 0)
    gt_area = np.sum(gt_mask > 0)
    
    # 重叠分析
    pred_gt_overlap = np.sum((pred_mask > 0) & (gt_mask > 0))
    pred_affordance_overlap = np.sum((pred_mask > 0) & (red_mask > 0))
    gt_affordance_overlap = np.sum((gt_mask > 0) & (red_mask > 0))
    
    # 计算IoU和覆盖率
    pred_gt_union = np.sum((pred_mask > 0) | (gt_mask > 0))
    pred_gt_iou = pred_gt_overlap / max(pred_gt_union, 1)
    
    pred_affordance_precision = pred_affordance_overlap / max(pred_area, 1)
    pred_affordance_recall = pred_affordance_overlap / max(affordance_area, 1)
    
    gt_affordance_precision = gt_affordance_overlap / max(gt_area, 1)  
    gt_affordance_recall = gt_affordance_overlap / max(affordance_area, 1)
    
    return {
        "areas": {
            "affordance": affordance_area,
            "object": object_area, 
            "pred": pred_area,
            "gt": gt_area
        },
        "overlaps": {
            "pred_gt": pred_gt_overlap,
            "pred_affordance": pred_affordance_overlap,
            "gt_affordance": gt_affordance_overlap
        },
        "metrics": {
            "pred_gt_iou": pred_gt_iou,
            "pred_affordance_precision": pred_affordance_precision,
            "pred_affordance_recall": pred_affordance_recall,
            "gt_affordance_precision": gt_affordance_precision,
            "gt_affordance_recall": gt_affordance_recall
        }
    }

def create_legend_image():
    """创建图例说明"""
    legend = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 绘制图例
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 颜色说明
    legend[20:40, 20:40] = [0, 80, 0]    # 深绿色
    cv2.putText(legend, "Object Areas (SAM)", (50, 35), font, font_scale, (255, 255, 255), thickness)
    
    legend[50:70, 20:40] = [0, 0, 80]    # 深红色  
    cv2.putText(legend, "Affordance Areas (SAM)", (50, 65), font, font_scale, (255, 255, 255), thickness)
    
    legend[80:100, 20:40] = [255, 255, 0]  # 青色
    cv2.putText(legend, "Predicted Points", (50, 95), font, font_scale, (255, 255, 255), thickness)
    
    legend[110:130, 20:40] = [0, 255, 255] # 黄色
    cv2.putText(legend, "Ground Truth Points", (50, 125), font, font_scale, (255, 255, 255), thickness)
    
    legend[140:160, 20:40] = [255, 255, 255] # 白色
    cv2.putText(legend, "Perfect Overlap", (50, 155), font, font_scale, (255, 255, 255), thickness)
    
    return legend

def print_analysis_summary(scene_id, frame_id, stats):
    """打印分析总结"""
    if stats is None:
        print(f"场景 {scene_id} 帧 {frame_id}: 无法分析")
        return
    
    areas = stats["areas"]
    metrics = stats["metrics"]
    
    print(f"\n=== 场景 {scene_id} 帧 {frame_id} 叠加分析 ===")
    print(f"📊 区域统计:")
    print(f"  Affordance区域: {areas['affordance']:,} 像素")
    print(f"  Object区域: {areas['object']:,} 像素") 
    print(f"  预测投影: {areas['pred']:,} 像素")
    print(f"  GT投影: {areas['gt']:,} 像素")
    
    print(f"🎯 质量指标:")
    print(f"  预测-GT IoU: {metrics['pred_gt_iou']:.3f}")
    print(f"  预测-Affordance精确度: {metrics['pred_affordance_precision']:.3f}")
    print(f"  预测-Affordance召回率: {metrics['pred_affordance_recall']:.3f}")
    print(f"  GT-Affordance精确度: {metrics['gt_affordance_precision']:.3f}")
    print(f"  GT-Affordance召回率: {metrics['gt_affordance_recall']:.3f}")
    
    # 质量判断
    if metrics['pred_gt_iou'] > 0.3:
        print("✅ 预测与GT重叠度较好")
    elif metrics['pred_gt_iou'] > 0.1:
        print("⚠️ 预测与GT重叠度中等")  
    else:
        print("❌ 预测与GT重叠度较差")
    
    if metrics['pred_affordance_recall'] > 0.5:
        print("✅ 预测很好地覆盖了affordance区域")
    elif metrics['pred_affordance_recall'] > 0.2:
        print("⚠️ 预测部分覆盖了affordance区域")
    else:
        print("❌ 预测几乎没有覆盖affordance区域")

def process_single_frame(scene_dir, frame_id, output_dir):
    """处理单个帧"""
    frame_dir = scene_dir / f"frame_{frame_id:04d}"
    if not frame_dir.exists():
        print(f"帧目录不存在: {frame_dir}")
        return False
    
    mask_path = frame_dir / "sam_masks.png"
    pred_path = frame_dir / "predicted_projection.png"  
    gt_path = frame_dir / "groundtruth_projection.png"
    
    # 创建输出目录
    output_frame_dir = output_dir / f"frame_{frame_id:04d}"
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建叠加可视化
    overlay_path = output_frame_dir / "overlay_analysis.png"
    success = create_overlay_visualization(mask_path, pred_path, gt_path, overlay_path)
    
    if success:
        # 分析统计
        stats = analyze_overlap_statistics(mask_path, pred_path, gt_path)
        print_analysis_summary(scene_dir.name, frame_id, stats)
        
        print(f"✅ 叠加可视化已保存: {overlay_path}")
        return True
    else:
        print(f"❌ 创建叠加可视化失败")
        return False

def main():
    parser = argparse.ArgumentParser(description="创建三种投影的叠加可视化")
    parser.add_argument("--scene_id", type=str, required=True, help="场景ID")
    parser.add_argument("--input_dir", type=str,
                       default="/nas/qirui/sam3/S2FUN/visualization_output_30scenes_affordance_only",
                       help="输入可视化目录")
    parser.add_argument("--output_dir", type=str,
                       default="/nas/qirui/sam3/S2FUN/overlay_analysis",
                       help="输出分析目录")
    parser.add_argument("--frame_ids", type=str, default="0,10,20,30",
                       help="要分析的帧ID，用逗号分隔")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scene_dir = input_dir / args.scene_id
    
    if not scene_dir.exists():
        print(f"❌ 场景目录不存在: {scene_dir}")
        return
    
    # 创建输出目录
    scene_output_dir = output_dir / args.scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_ids = [int(x.strip()) for x in args.frame_ids.split(",")]
    
    print(f"🔍 创建场景 {args.scene_id} 的叠加分析")
    print(f"📁 输入目录: {scene_dir}")
    print(f"📁 输出目录: {scene_output_dir}")
    
    success_count = 0
    for frame_id in frame_ids:
        if process_single_frame(scene_dir, frame_id, scene_output_dir):
            success_count += 1
    
    print(f"\n🎉 完成！成功处理 {success_count}/{len(frame_ids)} 个帧")
    print(f"📂 结果保存在: {scene_output_dir}")

if __name__ == "__main__":
    main()