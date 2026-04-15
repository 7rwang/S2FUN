#!/usr/bin/env python3

import time
from pathlib import Path
import re

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    output_dir = Path("/nas/qirui/sam3/S2FUN/visualization_output_30scenes")
    
    while True:
        if not output_dir.exists():
            print("输出目录尚未创建...")
            time.sleep(5)
            continue
            
        # 检查已处理的场景
        scene_dirs = [d for d in output_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d+", d.name)]
        scene_dirs = sorted(scene_dirs, key=lambda x: natural_key(x.name))
        
        print(f"=== 当前进展 ===")
        print(f"已开始处理的场景: {len(scene_dirs)}")
        
        for scene_dir in scene_dirs:
            frame_count = len([f for f in scene_dir.iterdir() if f.is_dir()])
            if frame_count > 0:
                # 检查最新的一帧是否有完整的3个文件
                latest_frames = sorted([f for f in scene_dir.iterdir() if f.is_dir()], 
                                     key=lambda x: natural_key(x.name))
                if latest_frames:
                    latest_frame = latest_frames[-1]
                    file_count = len(list(latest_frame.glob("*.png")))
                    status = "完成" if file_count == 3 else f"进行中({file_count}/3)"
                else:
                    status = "开始中"
            else:
                status = "开始中"
            print(f"  {scene_dir.name}: {frame_count} 帧, {status}")
        
        print(f"=== 等待5秒后更新 ===\n")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n监控结束")