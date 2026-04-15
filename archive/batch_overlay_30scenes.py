#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_overlay_30scenes.py

批量生成30个场景的overlay分析可视化
"""

import subprocess
import sys
from pathlib import Path
import re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    # 配置路径
    python_path = "/nas/qirui/.conda/envs/qirui/bin/python"
    script_path = "/nas/qirui/sam3/S2FUN/create_overlay_visualization.py"
    
    # 固定参数
    input_dir = "/nas/qirui/sam3/S2FUN/visualization_output_30scenes_affordance_only"
    output_dir = "/nas/qirui/sam3/S2FUN/overlay_analysis_30scenes"
    frame_ids = "0,10,20,30"  # 每10帧分析一次
    
    # 查找所有已处理的场景
    input_path = Path(input_dir)
    scene_dirs = []
    for item in input_path.iterdir():
        if item.is_dir() and re.fullmatch(r"\d+", item.name):
            scene_dirs.append(item)
    
    # 按数字顺序排序
    scene_dirs = sorted(scene_dirs, key=lambda x: natural_key(x.name))
    
    print(f"发现 {len(scene_dirs)} 个场景需要生成overlay分析:")
    for scene_dir in scene_dirs:
        print(f"  - {scene_dir.name}")
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 批量处理
    success_count = 0
    failed_scenes = []
    
    for i, scene_dir in enumerate(scene_dirs, 1):
        scene_id = scene_dir.name
        print(f"\n{'='*60}")
        print(f"生成场景 {scene_id} overlay分析 ({i}/{len(scene_dirs)})")
        print(f"{'='*60}")
        
        # 构建命令
        cmd = [
            python_path, script_path,
            "--scene_id", scene_id,
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--frame_ids", frame_ids
        ]
        
        try:
            # 执行overlay可视化
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ 场景 {scene_id} overlay分析生成成功")
                success_count += 1
            else:
                print(f"❌ 场景 {scene_id} overlay分析失败:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                failed_scenes.append((scene_id, f"返回码: {result.returncode}"))
                
        except subprocess.TimeoutExpired:
            print(f"❌ 场景 {scene_id} 超时 (>5分钟)")
            failed_scenes.append((scene_id, "处理超时"))
            
        except Exception as e:
            print(f"❌ 场景 {scene_id} 发生异常: {e}")
            failed_scenes.append((scene_id, f"异常: {e}"))
    
    # 总结报告
    print(f"\n{'='*60}")
    print(f"Overlay分析批量处理完成!")
    print(f"{'='*60}")
    print(f"总场景数: {len(scene_dirs)}")
    print(f"成功处理: {success_count}")
    print(f"失败场景: {len(failed_scenes)}")
    
    if failed_scenes:
        print(f"\n失败的场景:")
        for scene_id, reason in failed_scenes:
            print(f"  - {scene_id}: {reason}")
    
    print(f"\n结果保存在: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()