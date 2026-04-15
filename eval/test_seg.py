"""
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 test_seg.py \
    --data-root /nas/qirui/scenefun3d/data \
    --scene-id 420683 \
    --output-root /nas/qirui/sam3/scenefun3d_ex \
    --annot2prompt /nas/qirui/sam3/scenefun3d_ex/_prompts/420683_annot2prompt.json \
    --save-json /nas/qirui/sam3/scenefun3d_ex/seg_test_out/420683
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from seg import seg_scene


def _safe_read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scene-id", required=True, type=int)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--annot2prompt", required=True, help="Path to annot2prompt.json")
    ap.add_argument("--image-subdir", default="hires_wide")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompt-delim", default=";")
    ap.add_argument("--save-json", default=None, help="Optional path to save returned annot2bestmask.json")
    args = ap.parse_args()

    annot2prompt = _safe_read_json(Path(args.annot2prompt))

    out = seg_scene(
        data_root=args.data_root,
        output_root=args.output_root,
        scene_id=int(args.scene_id),
        annot2prompt=annot2prompt,
        image_subdir=args.image_subdir,
        thr=float(args.thr),
        save_vis=bool(args.save_vis),
        device=args.device,
        prompt_delim=args.prompt_delim,
    )

    if out is not None:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        # Non-rank0 processes return None under torchrun.
        if os.environ.get("RANK", "0") != "0":
            return


if __name__ == "__main__":
    main()
