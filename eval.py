#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py

Key updates:
1) --oracle-best-node mode DOES NOT require --pred-json (nor descriptions mapping).
2) Support multi-scene evaluation with Micro-average (global per-instance weighting):
   - Concatenate per-annot AP/AR/IoU values across scenes, then compute metrics once.
3) Node->GT overlap success rate (filtered by node type):
   For each predicted node 3D mask (default only type=affordance):
     overlap_ratio = max_j |pred ∩ gt_j| / |pred|
   success if overlap_ratio >= --node-overlap-thr (default 0.8)
4) NEW: --pred-ids-file supports manual node-id predictions (no pred-json needed in normal mode):
   - Format A (aligned): one integer per line (may be -1), aligned with descriptions.json order.
   - Format B (by-instruction): "instruction<TAB>node_id" or "instruction,node_id" per line (may be -1).

Notes
-----
- {scene_id} placeholder is supported in --pred-json / --pred-ids-file / --index-json / --gt-annotations / --descriptions-json.
- Micro-average here means: each GT instance (annot) has equal weight globally (big scenes contribute more instances).

Examples
--------
# (A) Single scene: oracle-best-node
python eval.py \
  --index-json /nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes/0.02/435324/scene_graph_point_indices.json \
  --data-root /nas/qirui/scenefun3d/data \
  --scene-id 435324 \
  --oracle-best-node \
  --node-overlap-thr 0.8 \
  --node-overlap-types affordance

# (B) Normal mode with manual ids file (aligned)
python eval.py \
  --pred-ids-file /nas/qirui/sam3/S2FUN/id.txt \
  --index-json /nas/qirui/sam3/S2FUN/eval_qwen/421254/scene_graph_point_indices.json \
  --data-root /nas/qirui/scenefun3d/data \
  --scene-id 421254 \
  --out-json /nas/qirui/sam3/S2FUN/421254/eval_manual_421254.json

# (C) Normal mode with manual ids file (by instruction)
python eval.py \
  --pred-ids-file /path/to/manual_pairs.txt \
  --index-json /nas/qirui/sam3/S2FUN/eval_qwen/421254/scene_graph_point_indices.json \
  --data-root /nas/qirui/scenefun3d/data \
  --scene-id 421254 \
  --out-json /path/to/eval_manual_421254.json

# (D) Multi-scene micro-average: oracle-best-node + scene-list
python eval.py \
  --index-json "/nas/qirui/sam3/scenefun3d_ex/Experiments4SceneGraph/experiment_6/{scene_id}/scene_graph_point_indices.json" \
  --data-root "/nas/qirui/scenefun3d/val" \
  --scene-list /nas/qirui/only_in_train_val.txt \
  --oracle-best-node \
  --node-overlap-thr 0.8 \
  --node-overlap-types affordance \
  --out-json "/nas/qirui/sam3/scenefun3d_ex/Experiments4SceneGraph/experiment_6/eval_oracle_micro_30scenes.json"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Small utils
# -------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _safe_read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _find_unique_json(scene_dir: Path, keyword: str) -> Path:
    """
    Find a unique json under scene_dir whose filename contains keyword.
    Example: keyword='descriptions' or 'annotations'
    """
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    cands: List[Path] = []
    for p in scene_dir.rglob("*.json"):
        if keyword.lower() in p.name.lower():
            cands.append(p)

    cands = sorted(cands, key=lambda x: _natural_key(str(x)))
    if len(cands) == 0:
        raise FileNotFoundError(f"No *{keyword}*.json found under {scene_dir}")
    return cands[0]


def _fmt_path(pattern: str, scene_id: str) -> str:
    return (pattern or "").replace("{scene_id}", str(scene_id))


def load_pred_ids_file(pred_ids_path: Path) -> Dict[str, Any]:
    """
    Supports:
      (A) numeric-only lines: each line is an int node_id (may be -1), aligned to descriptions order
      (B) per-line with instruction: 'instruction<TAB>node_id' or 'instruction,node_id'

    Returns:
      {"mode": "aligned", "ids": [int,...]}  OR
      {"mode": "by_instruction", "map": {norm_instruction: int_node_id}}
    """
    lines = pred_ids_path.read_text(encoding="utf-8").splitlines()
    pairs: List[Tuple[str, int]] = []
    ids: List[int] = []
    numeric_only = True

    for raw in lines:
        s = raw.strip()
        if not s:
            # empty => -1 (aligned)
            ids.append(-1)
            continue

        # numeric-only?
        try:
            v = int(s)
            ids.append(v)
            continue
        except Exception:
            numeric_only = False

        # split as instruction + id (comma or tab)
        if "\t" in s:
            ins, nid = s.rsplit("\t", 1)
        elif "," in s:
            ins, nid = s.rsplit(",", 1)
        else:
            raise ValueError(
                f"Invalid line in pred-ids-file (expected int, or 'instruction<TAB>id' / 'instruction,id'): {raw}"
            )

        ins = ins.strip()
        nid = nid.strip()
        if not ins:
            raise ValueError(f"Empty instruction in pred-ids-file line: {raw}")
        try:
            nid_i = int(nid)
        except Exception:
            raise ValueError(f"Invalid node id in pred-ids-file line: {raw}")

        pairs.append((_norm_text(ins), nid_i))

    if numeric_only and len(pairs) == 0:
        return {"mode": "aligned", "ids": ids}

    # prefer by-instruction mode if any pairs exist
    m: Dict[str, int] = {}
    for k, v in pairs:
        if k not in m:
            m[k] = int(v)
    return {"mode": "by_instruction", "map": m}


# -------------------------
# Descriptions / annotations (NORMAL mode only)
# -------------------------

def load_desc_to_annots(descriptions_json: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      desc2annots: desc_id -> [annot_id...]
      desc2text:   desc_id -> instruction text (best-effort)

    Expected raw format:
      {
        "descriptions": [
          {"desc_id": "...", "annot_id": "..."/[...], "description_prompt"/"original_prompt"/"text"/...: "..."},
          ...
        ]
      }
    """
    data = _safe_read_json(descriptions_json)
    desc2annots: Dict[str, List[str]] = {}
    desc2text: Dict[str, str] = {}

    for d in data.get("descriptions", []):
        desc_id = str(d.get("desc_id", "")).strip()
        ann = d.get("annot_id", [])

        if isinstance(ann, str):
            ann_list = [ann.strip()]
        elif isinstance(ann, list):
            ann_list = [str(x).strip() for x in ann if str(x).strip()]
        else:
            ann_list = []

        text = None
        for k in ["original_prompt", "description_prompt", "prompt", "text", "description", "sentence", "instruction"]:
            v = d.get(k, None)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break

        if desc_id:
            desc2annots[desc_id] = ann_list
            if text is not None:
                desc2text[desc_id] = text

    return desc2annots, desc2text


def build_instruction_to_desc_id(desc2text: Dict[str, str]) -> Dict[str, str]:
    """
    Build a normalized instruction -> desc_id mapping.
    If collisions occur, keep the first one (stable).
    """
    m: Dict[str, str] = {}
    for desc_id, text in desc2text.items():
        key = _norm_text(text)
        if key and key not in m:
            m[key] = desc_id
    return m


# -------------------------
# GT loading
# -------------------------

def load_gt_instances(annotations_json: Path, num_points: int) -> Tuple[List[np.ndarray], List[str]]:
    data = _safe_read_json(annotations_json)
    gt_masks: List[np.ndarray] = []
    gt_annot_ids: List[str] = []

    for a in data.get("annotations", []):
        annot_id = str(a.get("annot_id", "")).strip()
        idxs = a.get("indices", [])
        if not annot_id or not isinstance(idxs, list) or len(idxs) == 0:
            continue

        m = np.zeros((num_points,), dtype=bool)
        idxs_np = np.asarray(idxs, dtype=np.int64)
        idxs_np = idxs_np[(idxs_np >= 0) & (idxs_np < num_points)]
        if idxs_np.size == 0:
            continue
        m[idxs_np] = True

        gt_masks.append(m)
        gt_annot_ids.append(annot_id)

    return gt_masks, gt_annot_ids


def pred_indices_to_mask(indices: List[int], num_points: int) -> np.ndarray:
    m = np.zeros((num_points,), dtype=bool)
    if not indices:
        return m
    idx = np.asarray(indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < num_points)]
    if idx.size == 0:
        return m
    m[idx] = True
    return m


# -------------------------
# Metrics (fun3du style)
# -------------------------

def _ap_ar_iou_from_masks(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float, float]:
    tp = int(np.logical_and(gt_mask, pred_mask).sum())
    pred_pos = int(pred_mask.sum())
    gt_pos = int(gt_mask.sum())

    ap = float(tp / pred_pos) if pred_pos > 0 else 0.0
    ar = float(tp / gt_pos) if gt_pos > 0 else 0.0

    uni = gt_pos + pred_pos - tp
    iou = float(tp / uni) if uni > 0 else 0.0
    return ap, ar, iou


def _mean_recall(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    mat = (values[:, None] >= thresholds[None, :]).astype(np.float32)  # (N, T)
    return mat.mean(axis=1)


def fun3du_metrics_from_values(ap_vals: np.ndarray, ar_vals: np.ndarray, iou_vals: np.ndarray) -> Dict[str, float]:
    if ap_vals.size == 0:
        return {
            "mAP": 0.0, "AP50": 0.0, "AP25": 0.0,
            "mAR": 0.0, "AR50": 0.0, "AR25": 0.0,
            "mIoU": 0.0,
        }

    ap_th = np.linspace(0.5, 0.95, 10, dtype=np.float32)  # [0.50..0.95]
    map_i = _mean_recall(ap_vals.astype(np.float32), ap_th)
    mar_i = _mean_recall(ar_vals.astype(np.float32), ap_th)

    ap25 = (ap_vals >= 0.25).astype(np.float32).mean()
    ap50 = (ap_vals >= 0.50).astype(np.float32).mean()
    ar25 = (ar_vals >= 0.25).astype(np.float32).mean()
    ar50 = (ar_vals >= 0.50).astype(np.float32).mean()

    return {
        "mAP": float(map_i.mean()),
        "AP50": float(ap50),
        "AP25": float(ap25),
        "mAR": float(mar_i.mean()),
        "AR50": float(ar50),
        "AR25": float(ar25),
        "mIoU": float(iou_vals.mean()),
    }


# -------------------------
# Node loading (with type/name)
# -------------------------

def load_nodes(index_json: Path) -> List[Dict[str, Any]]:
    """
    Expected node entry example:
      {"id": 1, "type":"affordance", "name":"door_handle", "inst_id":"000", "point_indices":[...]}
    """
    data = _safe_read_json(index_json)
    arr = data.get("node_point_indices", [])
    if not isinstance(arr, list):
        raise ValueError(f"index json missing list field node_point_indices: {index_json}")

    nodes: List[Dict[str, Any]] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        nid = it.get("id", None)
        idxs = it.get("point_indices", None)
        if not isinstance(nid, int) or not isinstance(idxs, list):
            continue
        nodes.append({
            "id": int(nid),
            "type": str(it.get("type", "")).strip().lower(),
            "name": str(it.get("name", "")).strip().lower(),
            "inst_id": str(it.get("inst_id", "")).strip(),
            "point_indices": idxs,
        })
    return nodes


def nodes_to_id2idx(nodes: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    return {int(n["id"]): n["point_indices"] for n in nodes if isinstance(n.get("point_indices", None), list)}


def infer_num_points(ann_json_path: Path, node2idx: Dict[int, List[int]]) -> int:
    max_gt = 0
    ann_data = _safe_read_json(ann_json_path)
    for a in ann_data.get("annotations", []):
        idxs = a.get("indices", [])
        if isinstance(idxs, list) and idxs:
            try:
                max_gt = max(max_gt, int(max(idxs)))
            except Exception:
                pass

    max_pred = 0
    for idxs in node2idx.values():
        if isinstance(idxs, list) and idxs:
            try:
                max_pred = max(max_pred, int(max(idxs)))
            except Exception:
                pass

    mx = max(max_gt, max_pred)
    return (mx + 1) if mx > 0 else 1


# -------------------------
# Node overlap success rate (filtered by type)
# -------------------------

def node_overlap_success_rate(
    gt_masks: List[np.ndarray],
    gt_annot_ids: List[str],
    nodes: List[Dict[str, Any]],
    num_points: int,
    thr: float = 0.8,
    allowed_types: Optional[List[str]] = None,
    head_k: int = 50,
) -> Dict[str, Any]:
    """
    For each predicted node mask (optionally filtered by node["type"]):
      overlap_ratio(node) = max_j |node ∩ gt_j| / |node|
    success if overlap_ratio >= thr.
    """
    allow = None
    if allowed_types is not None:
        allow = set([str(t).strip().lower() for t in allowed_types if str(t).strip()])

    gt_items: List[Tuple[str, np.ndarray]] = [
        (aid, m) for aid, m in zip(gt_annot_ids, gt_masks) if m is not None and m.size == num_points
    ]

    node_ids_head: List[int] = []
    node_types_head: List[str] = []
    node_names_head: List[str] = []
    best_gt_ids_head: List[Optional[str]] = []
    best_overlap_head: List[float] = []

    succ = 0
    valid = 0
    filtered_out = 0

    for n in nodes:
        ntype = str(n.get("type", "")).strip().lower()
        if allow is not None and ntype not in allow:
            filtered_out += 1
            continue

        nid = n.get("id", None)
        idxs = n.get("point_indices", None)
        if not isinstance(nid, int) or not isinstance(idxs, list) or len(idxs) == 0:
            continue

        idx_arr = np.asarray(idxs, dtype=np.int64)
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < num_points)]
        if idx_arr.size == 0:
            continue

        valid += 1
        pred_sz = int(idx_arr.size)

        best_r = 0.0
        best_aid: Optional[str] = None

        for aid, gt_m in gt_items:
            tp = int(gt_m[idx_arr].sum())
            if tp == 0:
                continue
            r = float(tp / pred_sz)
            if r > best_r:
                best_r = r
                best_aid = aid

        if best_r >= thr:
            succ += 1

        if len(node_ids_head) < head_k:
            node_ids_head.append(int(nid))
            node_types_head.append(ntype)
            node_names_head.append(str(n.get("name", "")).strip().lower())
            best_gt_ids_head.append(best_aid)
            best_overlap_head.append(float(best_r))

    rate = float(succ / valid) if valid > 0 else 0.0
    return {
        "thr": float(thr),
        "allowed_types": sorted(list(allow)) if allow is not None else None,
        "num_nodes_filtered_out_by_type": int(filtered_out),
        "num_nodes_nonempty": int(valid),
        "num_nodes_success": int(succ),
        "success_rate": float(rate),
        "node_ids_head": node_ids_head,
        "node_types_head": node_types_head,
        "node_names_head": node_names_head,
        "best_gt_ids_head": best_gt_ids_head,
        "best_overlap_head": best_overlap_head,
    }


# -------------------------
# Per-scene evaluation
# -------------------------

def eval_one_scene_oracle_best_node(
    ann_json_path: Path,
    index_json_path: Path,
    node_overlap_thr: float,
    node_overlap_types: List[str],
) -> Dict[str, Any]:
    """
    Oracle: For each GT annot, choose the node with max IoU.
    Also computes node overlap success rate for filtered node types.
    """
    nodes = load_nodes(index_json_path)
    node2idx = nodes_to_id2idx(nodes)
    num_points = infer_num_points(ann_json_path, node2idx)

    gt_masks, gt_annot_ids = load_gt_instances(ann_json_path, num_points=num_points)

    # Pre-pack node indices as numpy arrays for fast gt_m[idx].sum()
    node_items: List[Tuple[int, np.ndarray]] = []
    for nid, idxs in node2idx.items():
        if not isinstance(idxs, list) or len(idxs) == 0:
            continue
        arr = np.asarray(idxs, dtype=np.int64)
        arr = arr[(arr >= 0) & (arr < num_points)]
        if arr.size == 0:
            continue
        node_items.append((int(nid), arr))

    best_node_ids: List[Optional[int]] = []
    best_ious: List[float] = []

    ap_vals: List[float] = []
    ar_vals: List[float] = []
    iou_vals: List[float] = []

    if len(node_items) == 0:
        for gt_m in gt_masks:
            ap, ar, iou = _ap_ar_iou_from_masks(gt_m, np.zeros_like(gt_m, dtype=bool))
            ap_vals.append(ap); ar_vals.append(ar); iou_vals.append(iou)
            best_node_ids.append(None); best_ious.append(0.0)
    else:
        for gt_m in gt_masks:
            gt_sum = int(gt_m.sum())
            if gt_sum == 0:
                ap_vals.append(0.0); ar_vals.append(0.0); iou_vals.append(0.0)
                best_node_ids.append(None); best_ious.append(0.0)
                continue

            best_iou = -1.0
            best_nid: Optional[int] = None
            best_idx_arr: Optional[np.ndarray] = None

            for nid, idx_arr in node_items:
                tp = int(gt_m[idx_arr].sum())
                if tp == 0:
                    continue
                pred_sum = int(idx_arr.size)
                uni = gt_sum + pred_sum - tp
                iou = float(tp / uni) if uni > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_nid = nid
                    best_idx_arr = idx_arr

            if best_nid is None or best_iou <= 0.0 or best_idx_arr is None:
                pred_m = np.zeros_like(gt_m, dtype=bool)
                best_node_ids.append(None)
                best_ious.append(0.0)
            else:
                pred_m = np.zeros_like(gt_m, dtype=bool)
                pred_m[best_idx_arr] = True
                best_node_ids.append(int(best_nid))
                best_ious.append(float(best_iou))

            ap, ar, iou = _ap_ar_iou_from_masks(gt_m, pred_m)
            ap_vals.append(ap); ar_vals.append(ar); iou_vals.append(iou)

    ap_arr = np.asarray(ap_vals, dtype=np.float32)
    ar_arr = np.asarray(ar_vals, dtype=np.float32)
    iou_arr = np.asarray(iou_vals, dtype=np.float32)

    metrics = fun3du_metrics_from_values(ap_arr, ar_arr, iou_arr)

    node_overlap = node_overlap_success_rate(
        gt_masks=gt_masks,
        gt_annot_ids=gt_annot_ids,
        nodes=nodes,
        num_points=num_points,
        thr=float(node_overlap_thr),
        allowed_types=node_overlap_types,
    )

    debug = {
        "oracle_best_node_enabled": True,
        "num_nodes_considered": int(len(node_items)),
        "oracle_mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "oracle_median_best_iou": float(np.median(best_ious)) if best_ious else 0.0,
        "oracle_best_node_ids_head50": best_node_ids[:50],
    }

    return {
        "metrics": metrics,
        "ap_vals": ap_arr,
        "ar_vals": ar_arr,
        "iou_vals": iou_arr,
        "gt_instances": int(len(gt_masks)),
        "num_points_used": int(num_points),
        "oracle_debug": debug,
        "node_overlap": node_overlap,
    }


def eval_one_scene_normal(
    pred_json_path: Optional[Path],
    pred_ids_path: Optional[Path],
    ann_json_path: Path,
    desc_json_path: Path,
    index_json_path: Path,
    use_best_of_k: bool,
    node_overlap_thr: float,
    node_overlap_types: List[str],
) -> Dict[str, Any]:
    """
    Normal:
      - pred-json: candidate_ids json + instruction->desc->annot mapping.
      - OR pred-ids-file: manual node_id per instruction (aligned or by instruction).
    Also computes node overlap success rate for filtered node types.
    """
    # descriptions mapping
    desc2annots, desc2text = load_desc_to_annots(desc_json_path)
    instr2desc = build_instruction_to_desc_id(desc2text)

    nodes = load_nodes(index_json_path)
    node2idx = nodes_to_id2idx(nodes)
    num_points = infer_num_points(ann_json_path, node2idx)

    # Load GT
    gt_masks, gt_annot_ids = load_gt_instances(ann_json_path, num_points=num_points)
    gt_by_annot: Dict[str, np.ndarray] = {aid: m for aid, m in zip(gt_annot_ids, gt_masks)}

    # node overlap success rate (independent of candidate matching)
    node_overlap = node_overlap_success_rate(
        gt_masks=gt_masks,
        gt_annot_ids=gt_annot_ids,
        nodes=nodes,
        num_points=num_points,
        thr=float(node_overlap_thr),
        allowed_types=node_overlap_types,
    )

    pred_by_annot: Dict[str, Dict[str, Any]] = {}

    missing_instr = 0
    missing_desc = 0
    missing_node = 0

    used_pred_ids_file = bool(pred_ids_path is not None and str(pred_ids_path).strip())

    # ------------------------------------------------------------
    # (1) Manual pred ids file (preferred if provided)
    # ------------------------------------------------------------
    if used_pred_ids_file:
        info = load_pred_ids_file(pred_ids_path)

        if info["mode"] == "aligned":
            ids: List[int] = info["ids"]

            # Use the raw descriptions list order from descriptions.json
            desc_raw = _safe_read_json(desc_json_path).get("descriptions", [])
            if not isinstance(desc_raw, list):
                raise ValueError(f"descriptions.json malformed: {desc_json_path}")

            L = min(len(desc_raw), len(ids))

            for i in range(L):
                d = desc_raw[i]
                if not isinstance(d, dict):
                    continue
                desc_id = str(d.get("desc_id", "")).strip()
                if not desc_id:
                    continue

                annot_ids = desc2annots.get(desc_id, [])
                if not annot_ids:
                    continue

                nid = int(ids[i])
                if nid < 0:
                    # no prediction
                    for annot_id in annot_ids:
                        if annot_id:
                            pred_by_annot[annot_id] = {"indices": [], "candidate_id": None, "desc_id": desc_id}
                    continue

                if nid not in node2idx:
                    missing_node += 1
                    for annot_id in annot_ids:
                        if annot_id:
                            pred_by_annot[annot_id] = {"indices": [], "candidate_id": nid, "desc_id": desc_id}
                    continue

                for annot_id in annot_ids:
                    if not annot_id:
                        continue
                    pred_by_annot[annot_id] = {
                        "indices": node2idx.get(nid, []),
                        "candidate_id": nid,
                        "desc_id": desc_id,
                    }

        else:
            # by instruction string
            ins_map: Dict[str, int] = info["map"]

            for ins_norm, nid in ins_map.items():
                desc_id = instr2desc.get(ins_norm, None)
                if desc_id is None:
                    missing_instr += 1
                    continue
                if desc_id not in desc2annots:
                    missing_desc += 1
                    continue

                annot_ids = desc2annots.get(desc_id, [])
                if not annot_ids:
                    continue

                nid_i = int(nid)
                if nid_i < 0:
                    for annot_id in annot_ids:
                        if annot_id:
                            pred_by_annot[annot_id] = {"indices": [], "candidate_id": None, "desc_id": desc_id}
                    continue

                if nid_i not in node2idx:
                    missing_node += 1
                    for annot_id in annot_ids:
                        if annot_id:
                            pred_by_annot[annot_id] = {"indices": [], "candidate_id": nid_i, "desc_id": desc_id}
                    continue

                for annot_id in annot_ids:
                    if not annot_id:
                        continue
                    pred_by_annot[annot_id] = {
                        "indices": node2idx.get(nid_i, []),
                        "candidate_id": nid_i,
                        "desc_id": desc_id,
                    }

    # ------------------------------------------------------------
    # (2) Fallback to pred-json
    # ------------------------------------------------------------
    else:
        if pred_json_path is None:
            raise ValueError("Normal mode requires --pred-json or --pred-ids-file.")
        pred = _safe_read_json(pred_json_path)
        results = pred.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"pred-json results must be a list: {pred_json_path}")

        for r in results:
            if not isinstance(r, dict):
                continue

            instruction = (r.get("instruction", "") or "").strip()
            if not instruction:
                continue

            cand_ids = r.get("candidate_ids", [])
            if not isinstance(cand_ids, list):
                cand_ids = []
            cand_ids = [int(x) for x in cand_ids if isinstance(x, int)]

            desc_id = instr2desc.get(_norm_text(instruction), None)
            if desc_id is None:
                missing_instr += 1
                continue
            if desc_id not in desc2annots:
                missing_desc += 1
                continue

            annot_ids = desc2annots.get(desc_id, [])
            if not annot_ids:
                continue

            # build candidate masks
            cand_masks: List[Tuple[int, np.ndarray]] = []
            for cid in cand_ids:
                idxs = node2idx.get(cid, None)
                if idxs is None:
                    continue
                cand_masks.append((cid, pred_indices_to_mask(idxs, num_points=num_points)))

            if not cand_masks:
                missing_node += 1
                for annot_id in annot_ids:
                    if not annot_id:
                        continue
                    pred_by_annot[annot_id] = {"indices": [], "candidate_id": None, "desc_id": desc_id}
                continue

            for annot_id in annot_ids:
                if not annot_id:
                    continue

                if use_best_of_k and annot_id in gt_by_annot:
                    gt_m = gt_by_annot[annot_id]
                    best_iou = -1.0
                    best_cid = None
                    for cid, pm in cand_masks:
                        _, _, iou = _ap_ar_iou_from_masks(gt_m, pm)
                        if iou > best_iou:
                            best_iou = iou
                            best_cid = cid

                    pred_by_annot[annot_id] = {
                        "indices": node2idx.get(best_cid, []) if best_cid is not None else [],
                        "candidate_id": best_cid,
                        "desc_id": desc_id,
                    }
                else:
                    cid0, _ = cand_masks[0]
                    pred_by_annot[annot_id] = {
                        "indices": node2idx.get(cid0, []),
                        "candidate_id": cid0,
                        "desc_id": desc_id,
                    }

    # compute per-instance values aligned to GT order
    ap_vals: List[float] = []
    ar_vals: List[float] = []
    iou_vals: List[float] = []

    pred_nonempty = 0
    covered = 0

    for annot_id, gt_m in zip(gt_annot_ids, gt_masks):
        entry = pred_by_annot.get(annot_id)
        if entry is None:
            pred_m = np.zeros_like(gt_m, dtype=bool)
        else:
            covered += 1
            pred_m = pred_indices_to_mask(entry.get("indices", []), num_points=num_points)

        if int(pred_m.sum() > 0) == 1:
            pred_nonempty += 1

        apv, arv, iouv = _ap_ar_iou_from_masks(gt_m, pred_m)
        ap_vals.append(apv); ar_vals.append(arv); iou_vals.append(iouv)

    ap_arr = np.asarray(ap_vals, dtype=np.float32)
    ar_arr = np.asarray(ar_vals, dtype=np.float32)
    iou_arr = np.asarray(iou_vals, dtype=np.float32)

    metrics = fun3du_metrics_from_values(ap_arr, ar_arr, iou_arr)

    return {
        "metrics": metrics,
        "ap_vals": ap_arr,
        "ar_vals": ar_arr,
        "iou_vals": iou_arr,
        "gt_instances": int(len(gt_masks)),
        "pred_instances_nonempty": int(pred_nonempty),
        "pred_annot_covered": int(covered),
        "num_points_used": int(num_points),
        "debug": {
            "missing_instruction_match": int(missing_instr),
            "missing_desc_id_mapping": int(missing_desc),
            "missing_candidate_id_in_index_json": int(missing_node),
            "use_best_of_k_oracle": bool(use_best_of_k),
            "used_pred_ids_file": bool(used_pred_ids_file),
            "pred_ids_file_path": str(pred_ids_path) if pred_ids_path is not None else "",
            "pred_json_path": str(pred_json_path) if pred_json_path is not None else "",
        },
        "node_overlap": node_overlap,
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred-json", default="", help="candidate_ids json. Optional for --oracle-best-node.")
    ap.add_argument(
        "--pred-ids-file",
        default="",
        help="Manual predictions as node ids. "
             "Two formats supported:\n"
             "  (1) one integer per line (aligned with descriptions.json order)\n"
             "  (2) 'instruction<TAB>node_id' or 'instruction,node_id' per line\n"
             "Use -1 for 'no prediction'. Optional for --oracle-best-node."
    )

    ap.add_argument("--index-json", required=True, help="node_point_indices json")

    ap.add_argument("--gt-annotations", default="", help="GT annotations.json. If empty, auto-find under data_root/scene_id.")
    ap.add_argument("--descriptions-json", default="", help="descriptions.json. If empty, auto-find under data_root/scene_id. Only needed for normal mode.")

    ap.add_argument("--data-root", default="", help="Dataset root containing scene folders (for auto-find)")
    ap.add_argument("--scene-id", default="", help="Single scene/visit id")
    ap.add_argument("--scene-ids", nargs="*", default=[], help="Multiple scene ids for micro-average")
    ap.add_argument("--scene-list", default="", help="Text file with one scene_id per line")

    ap.add_argument("--out-json", default="", help="Optional output json path")

    ap.add_argument("--use-best-of-k", action="store_true", help="Normal mode only: oracle over candidate_ids (pick best IoU among K).")

    ap.add_argument("--oracle-best-node", action="store_true", help="Oracle over ALL nodes: pick best IoU node per GT annot. No pred-json needed.")

    ap.add_argument("--node-overlap-thr", type=float, default=0.8,
                    help="Threshold for node overlap success: overlap=max_j |pred∩gt_j|/|pred|.")

    ap.add_argument(
        "--node-overlap-types",
        nargs="*",
        default=["affordance"],
        help="Which node types to include in node_overlap stats. Default: affordance. "
             "Example: --node-overlap-types affordance contextual",
    )

    args = ap.parse_args()

    # Resolve scene ids
    scenes: List[str] = []
    if args.scene_id:
        scenes.append(str(args.scene_id))
    if args.scene_ids:
        scenes.extend([str(x) for x in args.scene_ids if str(x).strip()])
    if args.scene_list:
        p = Path(args.scene_list)
        if not p.exists():
            raise FileNotFoundError(f"--scene-list not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                scenes.append(s)

    seen = set()
    scenes_uniq: List[str] = []
    for s in scenes:
        if s not in seen:
            scenes_uniq.append(s)
            seen.add(s)

    if len(scenes_uniq) == 0:
        raise ValueError("Provide --scene-id or --scene-ids or --scene-list.")

    if not args.oracle_best_node and (not args.pred_json) and (not args.pred_ids_file):
        raise ValueError("Normal mode requires --pred-json or --pred-ids-file. (For oracle upper bound, pass --oracle-best-node.)")

    if (not args.gt_annotations or (not args.descriptions_json and not args.oracle_best_node)) and not args.data_root:
        raise ValueError("Need --data-root for auto-find of annotations/descriptions (unless you pass explicit paths).")

    node_overlap_types = [str(x).strip().lower() for x in (args.node_overlap_types or []) if str(x).strip()]

    per_scene_outputs: Dict[str, Any] = {}

    # Global micro buffers for AP/AR/IoU
    all_ap: List[np.ndarray] = []
    all_ar: List[np.ndarray] = []
    all_iou: List[np.ndarray] = []

    total_gt = 0
    total_pred_nonempty = 0
    total_covered = 0

    # Global micro (count-based) for node-overlap
    global_node_valid = 0
    global_node_succ = 0

    for scene_id in scenes_uniq:
        index_json_path = Path(_fmt_path(args.index_json, scene_id))

        # resolve annotations
        if args.gt_annotations:
            ann_json_path = Path(_fmt_path(args.gt_annotations, scene_id))
        else:
            ann_json_path = _find_unique_json(Path(args.data_root) / str(scene_id), "annotations")

        if args.oracle_best_node:
            out = eval_one_scene_oracle_best_node(
                ann_json_path=ann_json_path,
                index_json_path=index_json_path,
                node_overlap_thr=float(args.node_overlap_thr),
                node_overlap_types=node_overlap_types,
            )
            per_scene_outputs[scene_id] = {
                "annotations_json": str(ann_json_path),
                "index_json": str(index_json_path),
                "oracle_best_node_metrics": out["metrics"],
                "oracle_best_node_debug": out["oracle_debug"],
                "gt_instances": out["gt_instances"],
                "num_points_used": out["num_points_used"],
                "node_overlap": out["node_overlap"],
            }
        else:
            if args.descriptions_json:
                desc_json_path = Path(_fmt_path(args.descriptions_json, scene_id))
            else:
                desc_json_path = _find_unique_json(Path(args.data_root) / str(scene_id), "descriptions")

            pred_json_path = Path(_fmt_path(args.pred_json, scene_id)) if args.pred_json else None
            pred_ids_path = Path(_fmt_path(args.pred_ids_file, scene_id)) if args.pred_ids_file else None

            out = eval_one_scene_normal(
                pred_json_path=pred_json_path,
                pred_ids_path=pred_ids_path,
                ann_json_path=ann_json_path,
                desc_json_path=desc_json_path,
                index_json_path=index_json_path,
                use_best_of_k=bool(args.use_best_of_k),
                node_overlap_thr=float(args.node_overlap_thr),
                node_overlap_types=node_overlap_types,
            )
            per_scene_outputs[scene_id] = {
                "pred_json": str(pred_json_path) if pred_json_path is not None else "",
                "pred_ids_file": str(pred_ids_path) if pred_ids_path is not None else "",
                "descriptions_json": str(desc_json_path),
                "annotations_json": str(ann_json_path),
                "index_json": str(index_json_path),
                "metrics": out["metrics"],
                "gt_instances": out["gt_instances"],
                "pred_instances_nonempty": out["pred_instances_nonempty"],
                "pred_annot_covered": out["pred_annot_covered"],
                "num_points_used": out["num_points_used"],
                "debug": out["debug"],
                "node_overlap": out["node_overlap"],
            }
            total_pred_nonempty += int(out["pred_instances_nonempty"])
            total_covered += int(out["pred_annot_covered"])

        # micro buffers for AP/AR/IoU
        all_ap.append(out["ap_vals"])
        all_ar.append(out["ar_vals"])
        all_iou.append(out["iou_vals"])
        total_gt += int(out["gt_instances"])

        # micro buffers for node overlap success (count-based)
        node_overlap = out.get("node_overlap", {})
        global_node_valid += int(node_overlap.get("num_nodes_nonempty", 0))
        global_node_succ += int(node_overlap.get("num_nodes_success", 0))

    ap_cat = np.concatenate(all_ap, axis=0) if all_ap else np.zeros((0,), dtype=np.float32)
    ar_cat = np.concatenate(all_ar, axis=0) if all_ar else np.zeros((0,), dtype=np.float32)
    iou_cat = np.concatenate(all_iou, axis=0) if all_iou else np.zeros((0,), dtype=np.float32)

    micro_metrics = fun3du_metrics_from_values(ap_cat, ar_cat, iou_cat)

    global_node_rate = float(global_node_succ / global_node_valid) if global_node_valid > 0 else 0.0
    global_node_overlap = {
        "thr": float(args.node_overlap_thr),
        "allowed_types": node_overlap_types,
        "num_nodes_nonempty": int(global_node_valid),
        "num_nodes_success": int(global_node_succ),
        "success_rate": float(global_node_rate),
    }

    output: Dict[str, Any] = {
        "mode": "oracle_best_node" if args.oracle_best_node else "normal",
        "scenes": scenes_uniq,
        "micro_average_metrics": micro_metrics,
        "total_gt_instances": int(total_gt),
        "micro_node_overlap": global_node_overlap,
        "per_scene": per_scene_outputs,
    }

    if not args.oracle_best_node:
        output["total_pred_instances_nonempty"] = int(total_pred_nonempty)
        output["total_pred_annot_covered"] = int(total_covered)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()