import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 你的主脚本：确保这里文件名/模块名对得上
from proj_3d import extract_annot_pointclouds


def load_best_masks_json(j: Dict[str, Any]) -> List[Tuple[str, int, Path]]:
    """
    Return list of (annot_id, frame_idx, mask_path).
    """
    out = []
    for k, v in j.items():
        annot_id = v.get("annot_id", k)
        frame = int(v["frame"])
        mask_png = Path(v["mask_png"])
        out.append((annot_id, frame, mask_png))
    return out


def ensure_link(src: Path, dst: Path) -> None:
    """
    Create symlink dst -> src. If symlink not allowed, fallback to copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.symlink(str(src), str(dst))
    except Exception:
        # fallback: copy (slower, but always works)
        shutil.copy2(str(src), str(dst))


def read_ply_header_and_sample_lines(ply_path: Path, sample_n: int = 5):
    lines = ply_path.read_text(encoding="utf-8").splitlines()
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            end_idx = i
            break
    if end_idx is None:
        raise AssertionError(f"{ply_path} missing end_header")

    header = lines[: end_idx + 1]
    data = [ln.strip() for ln in lines[end_idx + 1 :] if ln.strip()]
    return header, data[:sample_n], len(data)


def assert_ply_is_red(ply_path: Path) -> None:
    header, sample, n_data = read_ply_header_and_sample_lines(ply_path, sample_n=10)

    # 1) header contains rgb fields
    header_txt = "\n".join(header)
    assert "property uchar red" in header_txt, f"{ply_path} missing 'property uchar red'"
    assert "property uchar green" in header_txt, f"{ply_path} missing 'property uchar green'"
    assert "property uchar blue" in header_txt, f"{ply_path} missing 'property uchar blue'"

    # 2) data lines end with '255 0 0'
    assert n_data > 0, f"{ply_path} has 0 vertices"
    bad = []
    for ln in sample:
        # expected: x y z 255 0 0
        if not ln.endswith("255 0 0"):
            bad.append(ln)
    assert not bad, f"{ply_path} has non-red sample lines: {bad[:3]}"


def main():
    # -------------------------
    best_json_path = Path("/nas/qirui/sam3/scenefun3d_ex/seg_test_out/420673/annot2bestmask.json")

    if not best_json_path.exists():
        raise FileNotFoundError(
            f"best_masks.json not found: {best_json_path}\n"
            f"你需要把你发的那段 JSON 存成文件，然后改这里的路径。"
        )

    j = json.loads(best_json_path.read_text(encoding="utf-8"))
    items = load_best_masks_json(j)

    # -------------------------
    # 真实数据根目录/scene_id
    # -------------------------
    data_root = Path("/nas/qirui/scenefun3d/data")
    scene_id = 420673

    # -------------------------
    # 临时 masks_root（重排成 masks_root/<frame>/<annot_id>.png）
    # -------------------------
    tmp_root = Path("/nas/qirui/sam3/scenefun3d_ex/seg_test_out/420673/_tmp_extract_test")
    masks_root = tmp_root / "masks_by_frame"
    out_dir = tmp_root / "annot_pcd_red"

    # 清理旧结果（避免你以为这次跑出来的其实是上次残留）
    if masks_root.exists():
        shutil.rmtree(masks_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    masks_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只挑一些样本也行（加速）。默认全测。
    # items = items[:5]

    print(f"[1/4] Building masks_root at: {masks_root}")
    missing = []
    for annot_id, frame_idx, mask_path in items:
        if not mask_path.exists():
            missing.append(str(mask_path))
            continue
        dst = masks_root / str(frame_idx) / f"{annot_id}.png"
        ensure_link(mask_path, dst)

    if missing:
        raise FileNotFoundError(f"Some mask files are missing, e.g.:\n" + "\n".join(missing[:10]))

    print(f"[2/4] Running extract_annot_pointclouds...")
    summary = extract_annot_pointclouds(
        data_root=data_root,
        scene_id=scene_id,
        masks_root=masks_root,
        out_dir=out_dir,
        depth_scale=1000.0,     # 你真实数据若不是毫米，改这里
        stride_pixels=1,
        traj_is_T_wc=True,
        save_meta=True,
        cache_depth=True,
    )

    # -------------------------
    # 验证输出：npz/ply存在 + ply全红
    # -------------------------
    print(f"[3/4] Validating outputs in: {out_dir}")

    per_annot = summary.get("per_annot", {})
    saved = 0
    checked_red = 0
    empty = 0

    for annot_id, info in per_annot.items():
        ply = info.get("ply", None)
        npz = info.get("npz", None)
        pts = int(info.get("points", 0))

        if ply is None or npz is None:
            empty += 1
            continue

        ply_path = Path(ply)
        npz_path = Path(npz)
        assert ply_path.exists(), f"Missing PLY: {ply_path}"
        assert npz_path.exists(), f"Missing NPZ: {npz_path}"
        assert pts > 0, f"{annot_id} points=0 but files exist? suspicious"

        # 检查红色
        assert_ply_is_red(ply_path)

        saved += 1
        checked_red += 1

    print(f"[4/4] DONE.")
    print(f"  annot_total: {summary.get('annot_ids_total')}")
    print(f"  annot_saved: {summary.get('annot_ids_saved')}")
    print(f"  checked_red_ply: {checked_red}")
    print(f"  empty_or_failed: {empty}")
    print(f"  out_dir: {out_dir}")
    print(f"  meta: {out_dir/'extract_meta.json'}")


if __name__ == "__main__":
    main()
