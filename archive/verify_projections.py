#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path

def analyze_image(img_path):
    """分析图像内容"""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"error": "Failed to load image"}
    
    # 转换为灰度图进行分析
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 统计非黑色像素
    non_zero_pixels = np.sum(gray > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    content_ratio = non_zero_pixels / total_pixels
    
    # 统计不同颜色的像素
    unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
    
    return {
        "size": img.shape,
        "non_zero_pixels": non_zero_pixels,
        "total_pixels": total_pixels, 
        "content_ratio": content_ratio,
        "unique_colors": unique_colors,
        "mean_intensity": gray.mean()
    }

def main():
    output_dir = Path("/nas/qirui/sam3/S2FUN/debug_test_final/435324")
    
    print("=== 验证预测点云投影效果 ===")
    
    # 检查多个帧
    for frame_dir in sorted(output_dir.glob("frame_*"))[:3]:
        print(f"\n--- {frame_dir.name} ---")
        
        for img_type in ["sam_masks", "predicted_projection", "groundtruth_projection"]:
            img_path = frame_dir / f"{img_type}.png"
            if img_path.exists():
                stats = analyze_image(img_path)
                print(f"{img_type}:")
                print(f"  内容比例: {stats.get('content_ratio', 0)*100:.1f}%")
                print(f"  颜色数量: {stats.get('unique_colors', 0)}")
                print(f"  平均亮度: {stats.get('mean_intensity', 0):.1f}")
                
                # 特别检查是否还是全黑
                if stats.get('content_ratio', 0) < 0.001:
                    print(f"  ⚠️ 警告: {img_type} 几乎全黑!")
                elif img_type == "predicted_projection" and stats.get('content_ratio', 0) > 0.01:
                    print(f"  ✅ 成功: {img_type} 有丰富内容!")
            else:
                print(f"{img_type}: 文件不存在")
    
    # 比较文件大小
    print(f"\n=== 文件大小对比 ===")
    frame0 = output_dir / "frame_0000"
    if frame0.exists():
        for img_type in ["sam_masks", "predicted_projection", "groundtruth_projection"]:
            img_path = frame0 / f"{img_type}.png"
            if img_path.exists():
                size_kb = img_path.stat().st_size / 1024
                print(f"{img_type}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()