#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path

def analyze_sam_mask_colors(mask_path):
    """分析SAM mask中的确切颜色值"""
    img = cv2.imread(str(mask_path))
    if img is None:
        print(f"无法加载: {mask_path}")
        return
    
    print(f"\n=== 分析 {mask_path.name} ===")
    print(f"图像尺寸: {img.shape}")
    
    # 找到非黑色像素
    non_zero = np.any(img > 0, axis=2)
    if not np.any(non_zero):
        print("图像全黑，没有mask内容")
        return
    
    # 分析所有唯一颜色
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    print(f"唯一颜色数: {len(unique_colors)}")
    
    for i, color in enumerate(unique_colors):
        b, g, r = color
        pixel_count = np.sum(np.all(img == color, axis=2))
        print(f"  颜色{i}: BGR({b:3d},{g:3d},{r:3d}) RGB({r:3d},{g:3d},{b:3d}) - {pixel_count:,}像素")
        
        # 判断是什么类型
        if b == 0 and g == 255 and r == 0:
            print(f"    → 绿色 (Object)")
        elif b == 0 and g == 0 and r == 255:
            print(f"    → 红色 (Affordance)")
        elif b == 0 and g == 0 and r == 0:
            print(f"    → 黑色 (背景)")
        else:
            print(f"    → 其他颜色")

def main():
    # 分析一个包含affordance的帧
    sam_mask_path = Path("/nas/qirui/sam3/S2FUN/visualization_output_30scenes_affordance_only/435324/frame_0200/sam_masks.png")
    
    if sam_mask_path.exists():
        analyze_sam_mask_colors(sam_mask_path)
    else:
        print(f"文件不存在: {sam_mask_path}")
    
    # 分析原始的affordance mask文件
    print("\n" + "="*50)
    print("分析原始affordance mask文件:")
    
    affordance_masks = [
        "/nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv/435324/masks/200/INT__door_handle__000__area3945.png",
        "/nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv/435324/masks/200/INT__switch__000__area2649.png"
    ]
    
    for mask_file in affordance_masks:
        mask_path = Path(mask_file)
        if mask_path.exists():
            analyze_sam_mask_colors(mask_path)

if __name__ == "__main__":
    main()