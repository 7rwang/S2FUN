from pathlib import Path
import json

from build_prompt import build_annotid_to_prompt_from_root


def main():
    data_root = Path("/nas/qirui/scenefun3d/data")
    csv_path = Path("/nas/qirui/sam3/scenefun3d_ex/parse_result.csv")
    scene_id = "420673"

    annot2prompt_all = build_annotid_to_prompt_from_root(
        data_root=data_root,
        csv_path=csv_path,
        out_json_path=None,
    )

    scene_dir = data_root / scene_id
    assert scene_dir.exists(), f"Scene dir not found: {scene_dir}"

    annot_json = [p for p in scene_dir.iterdir()
                  if p.is_file() and "annot" in p.name.lower() and p.suffix == ".json"]
    assert len(annot_json) == 1, f"Ambiguous annot json: {annot_json}"

    with annot_json[0].open("r", encoding="utf-8") as f:
        ann_data = json.load(f)

    # ✅ 正确字段：annot_id
    scene_annot_ids = {
        str(a.get("annot_id")).strip()
        for a in ann_data.get("annotations", [])
        if a.get("annot_id")
    }

    annot2prompt_scene = {
        aid: prompt
        for aid, prompt in annot2prompt_all.items()
        if aid in scene_annot_ids
    }

    print("=" * 80)
    print(f"Scene {scene_id} — annot_id → prompt")
    print(f"Total annots: {len(annot2prompt_scene)}")
    print("=" * 80)

    for aid, prompt in sorted(annot2prompt_scene.items(), key=lambda x: x[0]):
        print(f"[{aid}] {prompt}")

    print("=" * 80)


if __name__ == "__main__":
    main()
