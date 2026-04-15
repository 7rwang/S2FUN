#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA_VISIBLE_DEVICES=8 python run.py \
  --data-root /nas/qirui/scenefun3d/data \
  --csv-path /nas/qirui/sam3/scenefun3d_ex/parse_result.csv \
  --work-root /nas/qirui/sam3/scenefun3d_ex \
  --no-depth-consistency \
  --debug \
  --scene-id 420673

修正点（核心）：
1) --no-depth-consistency 不再用 vis_thres=None 这种错误方式；而是显式传 use_depth_filter=False
2) 修复 pred_instances_nonempty 的 NameError（原来引用了未定义的 pred_masks）
3) CLI 的 --scene-id 真正生效：传入 _call_build_prompt(scene_id=...)
4) eval_one_scene 不再返回 None（避免上游某处没判空导致 NoneType.items）；非rank0返回 skipped dict
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

sys.path.append(os.path.dirname(__file__))


# -------------------------
# Small utils
# -------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _get_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def _safe_read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _find_unique_json(scene_dir: Path, must_contain_key: str) -> Path:
    """
    Find a json file under scene_dir that likely matches descriptions/annotations.
    Prefer filenames containing the key; otherwise, content includes key.
    """
    cands = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if not cands:
        raise FileNotFoundError(f"No .json files under: {scene_dir}")

    # filename hint
    name_hits = [p for p in cands if must_contain_key in p.name.lower()]
    if len(name_hits) == 1:
        return name_hits[0]

    # content hint
    content_hits = []
    for p in cands:
        try:
            obj = _safe_read_json(p)
            if must_contain_key in obj:
                content_hits.append(p)
        except Exception:
            continue

    if len(content_hits) == 1:
        return content_hits[0]

    raise ValueError(
        f"Ambiguous json for key='{must_contain_key}' in {scene_dir}. "
        f"Candidates={sorted([x.name for x in cands])}, "
        f"name_hits={sorted([x.name for x in name_hits])}, "
        f"content_hits={sorted([x.name for x in content_hits])}"
    )


def _resolve_scene_jsons(data_root: Path, scene_id: Union[int, str]) -> Tuple[Path, Path]:
    """
    Returns: (descriptions_json, annotations_json)
    Must both exist and be unambiguous.
    """
    sid = str(scene_id)
    scene_dir = data_root / sid
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    desc_json = _find_unique_json(scene_dir, "descriptions")
    ann_json = _find_unique_json(scene_dir, "annotations")
    return desc_json, ann_json


def _load_laser_scan_num_points(scene_dir: Path, scene_id: Union[int, str]) -> int:
    """
    Load N from {visit_id}_laser_scan.ply.
    IMPORTANT: evaluation indices refer to THIS original point ordering.

    Prefers open3d, but can fall back to parsing the PLY header directly.
    """
    sid = str(scene_id)
    ply = scene_dir / f"{sid}_laser_scan.ply"
    if not ply.exists():
        hits = sorted(list(scene_dir.glob("*_laser_scan.ply")), key=lambda p: _natural_key(p.name))
        if not hits:
            raise FileNotFoundError(f"Cannot find laser scan ply under {scene_dir} (expected {ply.name})")
        ply = hits[0]

    try:
        import open3d as o3d  # type: ignore
        pcd = o3d.io.read_point_cloud(str(ply))
        n = int(np.asarray(pcd.points).shape[0])
        if n > 0:
            return n
    except Exception:
        pass

    # Fallback: parse PLY header for "element vertex N"
    try:
        with ply.open("rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                s = line.decode("ascii", errors="ignore").strip()
                if s.startswith("element vertex"):
                    parts = s.split()
                    if len(parts) == 3 and parts[2].isdigit():
                        n = int(parts[2])
                        if n > 0:
                            return n
                if s == "end_header":
                    break
    except Exception as e:
        raise RuntimeError(f"Failed to parse ply header: {ply}") from e

    raise RuntimeError(
        "open3d not available and could not parse vertex count from ply header: "
        f"{ply}"
    )


# -------------------------
# GT loading (annot indices)
# -------------------------

def load_gt_instances(
    annotations_json: Path,
    num_points: int,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Returns:
      gt_masks: list of (N,) bool
      gt_annot_ids: list of annot_id
      gt_labels: list of label string
    """
    data = _safe_read_json(annotations_json)
    gt_masks: List[np.ndarray] = []
    gt_annot_ids: List[str] = []
    gt_labels: List[str] = []

    for a in data.get("annotations", []):
        annot_id = str(a.get("annot_id", "")).strip()
        label = str(a.get("label", "")).strip()
        idxs = a.get("indices", [])
        if not annot_id or not isinstance(idxs, list) or len(idxs) == 0:
            continue

        m = np.zeros((num_points,), dtype=bool)
        idxs_np = np.asarray(idxs, dtype=np.int64)
        idxs_np = idxs_np[(idxs_np >= 0) & (idxs_np < num_points)]
        m[idxs_np] = True

        gt_masks.append(m)
        gt_annot_ids.append(annot_id)
        gt_labels.append(label)

    return gt_masks, gt_annot_ids, gt_labels


# -------------------------
# Prediction loading (pred indices)
# -------------------------

def pred_indices_to_mask(indices: List[int], num_points: int) -> np.ndarray:
    m = np.zeros((num_points,), dtype=bool)
    if not indices:
        return m
    idx = np.asarray(indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < num_points)]
    m[idx] = True
    return m


# -------------------------
# fun3du-style metrics (per-annot masks)
# -------------------------

def compute_3d_ap(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    tp = np.logical_and(gt_mask, pred_mask).sum()
    positives = pred_mask.sum()
    return float(tp / positives) if positives > 0 else 0.0


def compute_3d_ar(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    tp = np.logical_and(gt_mask, pred_mask).sum()
    positives = gt_mask.sum()
    return float(tp / positives) if positives > 0 else 0.0


def compute_mean_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = np.logical_and(gt_mask, pred_mask).sum()
    uni = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter / uni) if uni > 0 else 0.0


def compute_mean_recalls(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    matrix = (values[:, None] >= thresholds[None, :]).astype(np.float32)
    return matrix.mean(axis=1)


def compute_scores(values: np.ndarray, thresholds: List[float]) -> Dict[float, np.ndarray]:
    return {th: (values >= th).astype(np.float32) for th in thresholds}


def fun3du_metrics(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
) -> Dict[str, float]:
    if len(gt_masks) == 0:
        return {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP25": 0.0,
            "mAR": 0.0,
            "AR50": 0.0,
            "AR25": 0.0,
            "mIoU": 0.0,
        }

    ap = np.array([compute_3d_ap(g, p) for g, p in zip(gt_masks, pred_masks)], dtype=np.float32)
    ar = np.array([compute_3d_ar(g, p) for g, p in zip(gt_masks, pred_masks)], dtype=np.float32)
    iou = np.array([compute_mean_iou(g, p) for g, p in zip(gt_masks, pred_masks)], dtype=np.float32)

    ap_th = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    map_i = compute_mean_recalls(ap, ap_th)
    mar_i = compute_mean_recalls(ar, ap_th)

    ap_rec = compute_scores(ap, [0.25, 0.50])
    ar_rec = compute_scores(ar, [0.25, 0.50])

    return {
        "mAP": float(map_i.mean()),
        "AP50": float(ap_rec[0.50].mean()),
        "AP25": float(ap_rec[0.25].mean()),
        "mAR": float(mar_i.mean()),
        "AR50": float(ar_rec[0.50].mean()),
        "AR25": float(ar_rec[0.25].mean()),
        "mIoU": float(iou.mean()),
    }


# -------------------------
# Adapters: build_prompt / seg
# -------------------------

def _call_build_prompt(
    data_root: Path,
    csv_path: Path,
    out_dir: Path,
    scene_id: str = "all",   # "all" or specific id string
    scene_id_filter: Optional[set] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      scene_id -> { annot_id -> prompt }
    """
    from build_prompt import build_annotid_to_prompt_from_root

    out_dir.mkdir(parents=True, exist_ok=True)

    # Global mapping across the (whole) CSV
    annot2prompt_global: Dict[str, str] = build_annotid_to_prompt_from_root(
        data_root=str(data_root),
        csv_path=str(csv_path),
        out_json_path=str(out_dir / "annot2prompt.json"),
    )

    # Collect scene_ids from CSV (respect filter + optional single-scene)
    scene_ids = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("scene_id", "")).strip()
            if not sid:
                continue
            if scene_id != "all" and sid != scene_id:
                continue
            if scene_id_filter is not None and sid not in scene_id_filter:
                continue
            scene_ids.add(sid)

    scene2map: Dict[str, Dict[str, str]] = {}

    for sid in sorted(scene_ids, key=lambda x: int(x) if x.isdigit() else x):
        desc_json, _ann_json = _resolve_scene_jsons(data_root, sid)
        desc_obj = _safe_read_json(desc_json)

        # all annot_ids in this scene
        scene_annots = set()
        for d in desc_obj.get("descriptions", []):
            ann = d.get("annot_id", [])
            if isinstance(ann, str):
                scene_annots.add(ann)
            elif isinstance(ann, list):
                for x in ann:
                    if x:
                        scene_annots.add(str(x))

        local = {aid: annot2prompt_global[aid] for aid in scene_annots if aid in annot2prompt_global}
        scene2map[sid] = local

        (out_dir / f"{sid}_annot2prompt.json").write_text(
            json.dumps(local, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return scene2map


def _call_seg(
    scene_id: Union[int, str],
    annot2prompt: Dict[str, str],
    data_root: Path,
    out_dir: Path,
    **seg_kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Calls seg.seg_scene(...) and returns:
      annot_id -> {mask_png, score, frame, ...}

    注意：你自己的 seg_scene 在多卡/多进程下可能非rank0返回 None。
    我们在 eval_one_scene 里兜底处理。
    """
    import seg  # your seg.py

    out_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(seg, "seg_scene"):
        raise AttributeError("seg.py must expose: seg_scene(...)")

    return seg.seg_scene(
        data_root=data_root,
        output_root=out_dir,
        scene_id=int(scene_id),
        annot2prompt=annot2prompt,
        **seg_kwargs,
    )


# -------------------------
# End-to-end: run one scene
# -------------------------

def eval_one_scene(
    data_root: Union[str, Path],
    scene_id: Union[int, str],
    annot2prompt: Dict[str, str],
    work_root: Union[str, Path],
    *,
    seg_kwargs: Optional[Dict[str, Any]] = None,
    vis_thres: float = 0.25,
    use_depth_filter: bool = True,
    save_artifacts: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Runs:
      seg -> laser projection -> fun3du-style eval

    始终返回 dict（不会返回 None），避免上游漏判空导致 NoneType.items。
    """
    data_root = Path(data_root)
    work_root = Path(work_root)
    sid = str(scene_id)
    rank = _get_rank()

    seg_kwargs = seg_kwargs or {}
    if debug:
        print(f"[STEP] scene_id={sid} rank={rank}")

    scene_dir = data_root / sid
    desc_json, ann_json = _resolve_scene_jsons(data_root, sid)

    # N points in original laser scan
    num_points = _load_laser_scan_num_points(scene_dir, sid)
    if debug:
        print(f"[STEP] num_points={num_points}")

    # GT masks
    gt_masks, gt_annot_ids, gt_labels = load_gt_instances(ann_json, num_points=num_points)
    if debug:
        print(f"[STEP] gt_instances={len(gt_masks)}")

    # --- SEG (best mask per annot_id) ---
    seg_out_dir = work_root / sid / "seg"
    # pred_json = _call_seg(
    #     scene_id=sid,
    #     annot2prompt=annot2prompt,
    #     data_root=data_root,
    #     out_dir=seg_out_dir,
    #     **seg_kwargs,
    # )
    # pred_json_path = Path("/nas/qirui/sam3/scenefun3d_ex/seg_test_out/420683.json")
    # pred_json = _safe_read_json(pred_json_path)
    from seg import seg_scene    
    out = seg_scene(
        data_root=data_root,
        output_root=work_root,          # ✅ 根目录
        scene_id=int(scene_id),
        annot2prompt=annot2prompt,
        device=device,                  # ✅ 你自己从args/seg_kwargs拿
    )

    pred_json = out

    if debug:
        print(f"[STEP] seg_out_dir={seg_out_dir}")
        print(f"[STEP] pred_json_type={type(pred_json)}")
        if isinstance(pred_json, dict):
            print(f"[STEP] pred_json_len={len(pred_json)}")
            if pred_json:
                k = next(iter(pred_json))
                print(f"[STEP] sample_annot_id={k}")
                try:
                    print(f"[STEP] sample_mask_png={pred_json[k].get('mask_png')}")
                except Exception:
                    pass

    # 多进程/多卡：有些实现会让非rank0返回 None
    if pred_json is None:
        return {
            "scene_id": sid,
            "rank": int(rank),
            "skipped": True,
            "reason": "pred_json is None (likely non-rank0 seg_scene return)",
            "paths": {
                "descriptions_json": str(desc_json),
                "annotations_json": str(ann_json),
                "seg_out_dir": str(seg_out_dir),
            },
        }

    if not isinstance(pred_json, dict):
        return {
            "scene_id": sid,
            "rank": int(rank),
            "skipped": True,
            "reason": f"pred_json is not dict: {type(pred_json)}",
        }

    # --- Laser projection to indices ---
    from eval_metric import pred_json_to_indices  # your corrected T_cw version

    pred3d = pred_json_to_indices(
        pred_json,
        data_root=data_root,
        scene_id=sid,
        masks_root=None,
        intr_dirname=seg_kwargs.get("intr_dirname", "hires_wide_intrinsics"),
        traj_name=seg_kwargs.get("traj_name", "hires_poses.traj"),
        depth_dirname=seg_kwargs.get("depth_dirname", "hires_depth"),
        mask_thr=seg_kwargs.get("mask_thr", 0.5),
        vis_thres=float(vis_thres),                 # 必须是 float
        use_depth_filter=bool(use_depth_filter),    # 显式控制是否用深度一致性
    )

    if debug:
        print(f"[STEP] pred3d_len={len(pred3d)}")

    # Convert pred3d -> per-annot masks aligned with GT annot_id
    gt_id_to_mask = {aid: gt_masks[i] for i, aid in enumerate(gt_annot_ids)}
    pred_masks_aligned: List[np.ndarray] = []
    gt_masks_aligned: List[np.ndarray] = []

    for aid, gt_m in gt_id_to_mask.items():
        gt_masks_aligned.append(gt_m)
        idxs = pred3d.get(aid, {}).get("indices", [])
        pred_masks_aligned.append(pred_indices_to_mask(idxs, num_points=num_points))

    metrics = fun3du_metrics(gt_masks=gt_masks_aligned, pred_masks=pred_masks_aligned)

    pred_nonempty = int(sum(int(m.sum() > 0) for m in pred_masks_aligned))

    if debug:
        print(f"[STEP] metrics={metrics}")
        print(f"[STEP] pred_instances_nonempty={pred_nonempty}")

    out = {
        "scene_id": sid,
        "rank": int(rank),
        "num_points": int(num_points),
        "gt_instances": int(len(gt_masks)),
        "pred_instances_nonempty": int(pred_nonempty),
        "metrics": {**metrics},
        "paths": {
            "descriptions_json": str(desc_json),
            "annotations_json": str(ann_json),
            "seg_out_dir": str(seg_out_dir),
        },
        "settings": {
            "vis_thres": float(vis_thres),
            "use_depth_filter": bool(use_depth_filter),
        }
    }

    if save_artifacts:
        (work_root / sid).mkdir(parents=True, exist_ok=True)
        (work_root / sid / "pred3d.json").write_text(
            json.dumps(pred3d, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (work_root / sid / "eval.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return out


# -------------------------
# End-to-end: run all scenes from CSV
# -------------------------

def run_eval(
    data_root: Union[str, Path],
    csv_path: Union[str, Path],
    work_root: Union[str, Path],
    *,
    seg_kwargs: Optional[Dict[str, Any]] = None,
    scene_id_csv: Optional[Union[str, Path]] = None,
    summary_out: Optional[Union[str, Path]] = None,
    scene_id: str = "all",
    vis_thres: float = 0.25,
    use_depth_filter: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline:
      build_prompt (per-scene)
      for each scene: seg -> laser projection -> fun3du-style eval
    """
    data_root = Path(data_root)
    csv_path = Path(csv_path)
    work_root = Path(work_root)
    rank = _get_rank()

    seg_kwargs = seg_kwargs or {}

    scene_id_filter = None
    if scene_id_csv is not None:
        scene_id_filter = set()
        with Path(scene_id_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "scene_id" not in reader.fieldnames:
                raise ValueError("scene_id.csv must contain a 'scene_id' column")
            for row in reader:
                sid = str(row.get("scene_id", "")).strip()
                if sid:
                    scene_id_filter.add(sid)

    # 1) build_prompt（这里必须传入 scene_id，保证 --scene-id 真生效）
    prompt_dir = work_root / "_prompts"
    scene2annot2prompt = _call_build_prompt(
        data_root=data_root,
        csv_path=csv_path,
        out_dir=prompt_dir,
        scene_id=scene_id,
        scene_id_filter=scene_id_filter,
    )

    results: List[Dict[str, Any]] = []
    for sid, annot2prompt in scene2annot2prompt.items():
        if not annot2prompt:
            continue

        r = eval_one_scene(
            data_root=data_root,
            scene_id=sid,
            annot2prompt=annot2prompt,
            work_root=work_root,
            seg_kwargs=seg_kwargs,
            vis_thres=vis_thres,
            use_depth_filter=use_depth_filter,
            save_artifacts=True,
            debug=debug,
        )

        # 只收集非 skipped 的结果用于 summary（你也可以选择保留 skipped）
        if r.get("skipped", False):
            if debug:
                print(f"[SKIP] scene={sid} reason={r.get('reason')}")
            continue

        results.append(r)

    # rank != 0 不做汇总
    if rank != 0:
        return {"summary": {}, "scenes": []}

    # aggregate
    if results:
        mAPs = [x["metrics"]["mAP"] for x in results]
        AP50s = [x["metrics"]["AP50"] for x in results]
        AP25s = [x["metrics"]["AP25"] for x in results]
        mARs = [x["metrics"]["mAR"] for x in results]
        AR50s = [x["metrics"]["AR50"] for x in results]
        AR25s = [x["metrics"]["AR25"] for x in results]
        mIoUs = [x["metrics"]["mIoU"] for x in results]
        summary = {
            "num_scenes": int(len(results)),
            "mAP_mean": float(np.mean(mAPs)),
            "AP50_mean": float(np.mean(AP50s)),
            "AP25_mean": float(np.mean(AP25s)),
            "mAR_mean": float(np.mean(mARs)),
            "AR50_mean": float(np.mean(AR50s)),
            "AR25_mean": float(np.mean(AR25s)),
            "mIoU_mean": float(np.mean(mIoUs)),
        }
    else:
        summary = {
            "num_scenes": 0,
            "mAP_mean": 0.0,
            "AP50_mean": 0.0,
            "AP25_mean": 0.0,
            "mAR_mean": 0.0,
            "AR50_mean": 0.0,
            "AR25_mean": 0.0,
            "mIoU_mean": 0.0,
        }

    out = {"summary": summary, "scenes": results}
    work_root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_out) if summary_out is not None else (work_root / "run_eval_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# -------------------------
# CLI entry
# -------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--scene-csv", default=None, help="CSV with scene_id column to filter scenes")
    ap.add_argument(
        "--scene-id",
        type=str,
        default="all",
        help="If specified, only run this scene_id instead of all scenes in CSV.",
    )
    ap.add_argument("--no-depth-consistency", action="store_true", help="Disable depth consistency check")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints")
    ap.add_argument(
        "--summary-out",
        default=None,
        help="Output path for run_eval summary JSON. Default: <work_root>/run_eval_summary.json",
    )

    # passthrough knobs (example)
    ap.add_argument("--thr", type=float, default=0.5, help="example seg threshold (passed to seg_kwargs)")
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    seg_kwargs = {
        "thr": float(args.thr),
        # 你也可以把 intr_dirname / traj_name / depth_dirname / mask_thr 放这里
        # "intr_dirname": "hires_wide_intrinsics",
        # "traj_name": "hires_poses.traj",
        # "depth_dirname": "hires_depth",
        # "mask_thr": 0.5,
    }

    # 正确的 depth consistency 控制方式：
    use_depth_filter = (not args.no_depth_consistency)
    vis_thres = 0.25  # 阈值永远是 float，不要用 None

    out = run_eval(
        data_root=args.data_root,
        csv_path=args.csv_path,
        work_root=args.work_root,
        seg_kwargs=seg_kwargs,
        scene_id_csv=args.scene_csv,
        summary_out=args.summary_out,
        scene_id=args.scene_id,
        vis_thres=vis_thres,
        use_depth_filter=use_depth_filter,
        debug=args.debug,
    )
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
