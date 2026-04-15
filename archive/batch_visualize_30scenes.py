#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_visualize_30scenes.py

批量生成30个场景的2D可视化
处理 /nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes 目录下的所有场景
"""

import subprocess
import sys
from pathlib import Path
import re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    # 配置路径
    eval_dir = Path("/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes")
    python_path = "/nas/qirui/.conda/envs/qirui/bin/python"
    script_path = "/nas/qirui/sam3/S2FUN/visualize_2d_scene.py"
    
    # 固定参数
    masks_root = "/nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv"
    data_root = "/nas/qirui/scenefun3d/data"
    gt_pcds_root = "/nas/qirui/scenefun3d/ann_pcds"
    output_dir = "/nas/qirui/sam3/S2FUN/visualization_output_30scenes_affordance_only"
    frame_stride = "10"  # 每10帧处理一次
    
    # 查找所有场景ID目录
    scene_dirs = []
    for item in eval_dir.iterdir():
        if item.is_dir() and re.fullmatch(r"\d+", item.name):
            scene_dirs.append(item)
    
    # 按数字顺序排序
    scene_dirs = sorted(scene_dirs, key=lambda x: natural_key(x.name))
    
    print(f"发现 {len(scene_dirs)} 个场景需要处理:")
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
        print(f"处理场景 {scene_id} ({i}/{len(scene_dirs)})")
        print(f"{'='*60}")
        
        # 检查scene_graph.json是否存在
        scene_graph_json = scene_dir / "scene_graph.json"
        if not scene_graph_json.exists():
            print(f"❌ 跳过场景 {scene_id}: scene_graph.json 不存在")
            failed_scenes.append((scene_id, "scene_graph.json不存在"))
            continue
        
        # 构建命令
        cmd = [
            python_path, script_path,
            "--scene_id", scene_id,
            "--masks_root", masks_root,
            "--data_root", data_root,
            "--scene_graph_json", str(scene_graph_json),
            "--gt_pcds_root", gt_pcds_root,
            "--output_dir", output_dir,
            "--frame_stride", frame_stride
        ]
        
        try:
            # 执行可视化
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ 场景 {scene_id} 处理成功")
                success_count += 1
            else:
                print(f"❌ 场景 {scene_id} 处理失败:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                failed_scenes.append((scene_id, f"返回码: {result.returncode}"))
                
        except subprocess.TimeoutExpired:
            print(f"❌ 场景 {scene_id} 超时 (>10分钟)")
            failed_scenes.append((scene_id, "处理超时"))
            
        except Exception as e:
            print(f"❌ 场景 {scene_id} 发生异常: {e}")
            failed_scenes.append((scene_id, f"异常: {e}"))
    
    # 总结报告
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
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