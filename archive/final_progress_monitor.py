#!/usr/bin/env python3

import time
from pathlib import Path

def main():
    output_dir = Path("/nas/qirui/sam3/S2FUN/visualization_output_30scenes_final")
    total_scenes = 30
    
    print("=== 最终批处理进度监控 ===")
    
    while True:
        if not output_dir.exists():
            print("输出目录尚未创建...")
            time.sleep(5)
            continue
            
        # 统计完成的场景
        scene_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        completed_count = len(scene_dirs)
        
        print(f"进度: {completed_count}/{total_scenes} ({completed_count/total_scenes*100:.1f}%)")
        
        if completed_count >= total_scenes:
            print("✅ 所有30个场景处理完成！")
            break
        
        # 显示最新完成的场景
        if scene_dirs:
            latest = sorted(scene_dirs)[-1]
            frame_count = len([f for f in latest.iterdir() if f.is_dir()])
            print(f"最新: {latest.name} ({frame_count} 帧)")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n监控结束")