#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rollback merge for scene_graph.json (post-processing).

CHANGED:
- Only merge within same "type"
- AND only merge when names are consistent:
    * node["name"] matches agg canonical name OR
    * intersection(node["names"], agg.names) is non-empty
  Otherwise: NEVER merge even if distance is small.

python rollback_merge_json.py \
--in_json /nas/qirui/sam3/scenefun3d_ex/420673/scene_graph.json \
--out_json /nas/qirui/sam3/scenefun3d_ex/420673/merge.json \
--mode l2 \
--thr 2.6 \
--grid_cell 0

python rollback_merge_json.py \
  --in_json /nas/qirui/sam3/scenefun3d_ex/420673/scene_graph.json \
  --mode l2 \
  --scan \
  --thr_list 2.0,2.1,2.2,2.3,2.4,2.5,2.6 \
  --grid_cell 0


"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np


# -------------------------
# Helpers
# -------------------------

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def as_np_pos(node: dict) -> np.ndarray:
    return np.array(node["position"], dtype=np.float64)

def union_names(a: List[str], b: List[str]) -> List[str]:
    s = set(a) | set(b)
    return sorted(s)

def ensure_names(node: dict) -> None:
    if "names" not in node or not isinstance(node["names"], list) or len(node["names"]) == 0:
        node["names"] = [node.get("name", "unknown")]

def canonical_name(names: List[str], fallback: str) -> str:
    if names:
        return names[0]
    return fallback

def names_overlap(a: List[str], b: List[str]) -> bool:
    """Return True if two name lists overlap (set intersection non-empty)."""
    return len(set(a) & set(b)) > 0

def name_compatible(node: dict, agg_names: List[str], agg_name: str) -> bool:
    """
    Decide whether a raw node can be merged into an agg by name.
    We allow merge if:
      - node["name"] == agg_name OR
      - intersection(node["names"], agg_names) != empty
    """
    n_name = node.get("name", "unknown")
    n_names = node.get("names", [n_name])
    if n_name == agg_name:
        return True
    return names_overlap(n_names, agg_names)


# -------------------------
# Distance functions
# -------------------------

@dataclass
class DistConfig:
    mode: str
    thr: float = 0.5

    # for weighted_l2
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # for axis
    axis_thr: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    axis_also_l2: Optional[float] = None  # if set: also require l2 < this

    # for hybrid
    thr_xy: float = 0.35
    thr_z: float = 0.20

    # for mahalanobis
    maha_eps: float = 1e-6  # covariance regularization


def dist_ok(p: np.ndarray, q: np.ndarray, cfg: DistConfig, inv_cov: Optional[np.ndarray]=None) -> bool:
    d = p - q

    if cfg.mode == "l2":
        return float(np.linalg.norm(d)) < cfg.thr

    if cfg.mode == "l2_xy":
        return float(np.linalg.norm(d[:2])) < cfg.thr

    if cfg.mode == "weighted_l2":
        sx, sy, sz = cfg.scale
        dd = np.array([d[0]/sx, d[1]/sy, d[2]/sz], dtype=np.float64)
        return float(np.linalg.norm(dd)) < cfg.thr

    if cfg.mode == "mahalanobis":
        if inv_cov is None:
            raise ValueError("mahalanobis requires inv_cov")
        val = float(np.sqrt(d.T @ inv_cov @ d))
        return val < cfg.thr

    if cfg.mode == "axis":
        tx, ty, tz = cfg.axis_thr
        ok = (abs(d[0]) < tx) and (abs(d[1]) < ty) and (abs(d[2]) < tz)
        if not ok:
            return False
        if cfg.axis_also_l2 is not None:
            return float(np.linalg.norm(d)) < cfg.axis_also_l2
        return True

    if cfg.mode == "hybrid":
        return (float(np.linalg.norm(d[:2])) < cfg.thr_xy) and (abs(d[2]) < cfg.thr_z)

    raise ValueError(f"Unknown mode: {cfg.mode}")


# -------------------------
# Grid neighbor search (acceleration)
# -------------------------

def grid_key(pos: np.ndarray, cell: float) -> Tuple[int, int, int]:
    return (int(np.floor(pos[0] / cell)),
            int(np.floor(pos[1] / cell)),
            int(np.floor(pos[2] / cell)))

def neighbor_keys(k: Tuple[int,int,int]) -> Iterable[Tuple[int,int,int]]:
    x,y,z = k
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                yield (x+dx, y+dy, z+dz)


# -------------------------
# Merge logic
# -------------------------

@dataclass
class AggNode:
    id: int
    type: str
    name: str
    names: List[str]
    pos_sum: np.ndarray
    pos_count: int
    merged_from: List[int] = field(default_factory=list)

    def position(self) -> np.ndarray:
        return self.pos_sum / max(1, self.pos_count)

    def absorb(self, other: "AggNode"):
        self.names = union_names(self.names, other.names)
        self.pos_sum += other.pos_sum
        self.pos_count += other.pos_count
        self.merged_from += other.merged_from


def build_inv_cov(positions: np.ndarray, maha_eps: float) -> np.ndarray:
    if positions.shape[0] < 2:
        return np.eye(3, dtype=np.float64)
    cov = np.cov(positions.T)
    cov = cov + np.eye(3, dtype=np.float64) * maha_eps
    return np.linalg.inv(cov)


def rollback_merge_nodes(
    nodes: List[dict],
    cfg: DistConfig,
    grid_cell: Optional[float] = None,
    keep_debug_fields: bool = False,
) -> List[dict]:
    """
    Merge nodes by cfg, within same type only,
    AND only if names are compatible (see name_compatible()).
    Uses grid acceleration if grid_cell is provided.
    """
    groups: Dict[str, List[dict]] = {}
    for n in nodes:
        ensure_names(n)
        groups.setdefault(n["type"], []).append(n)

    out_nodes: List[dict] = []
    next_id = 1

    for typ, arr in groups.items():
        inv_cov = None
        if cfg.mode == "mahalanobis":
            pos = np.stack([as_np_pos(x) for x in arr], axis=0)
            inv_cov = build_inv_cov(pos, cfg.maha_eps)

        cell = grid_cell
        if cell is None:
            if cfg.mode == "hybrid":
                cell = max(cfg.thr_xy, cfg.thr_z)
            elif cfg.mode == "axis":
                cell = max(cfg.axis_thr)
            else:
                cell = cfg.thr
            cell = max(1e-6, float(cell))

        grid: Dict[Tuple[int,int,int], List[int]] = {}
        aggs: List[AggNode] = []

        def add_to_grid(idx_agg: int):
            k = grid_key(aggs[idx_agg].position(), cell)
            grid.setdefault(k, []).append(idx_agg)

        for node in arr:
            pos = as_np_pos(node)
            if not np.isfinite(pos).all():
                continue

            cand = []
            k = grid_key(pos, cell)
            for kk in neighbor_keys(k):
                cand += grid.get(kk, [])

            best_i = None
            best_d = None

            for i_agg in cand:
                # --- NAME GATING (NEW) ---
                if not name_compatible(node, aggs[i_agg].names, aggs[i_agg].name):
                    continue

                p = aggs[i_agg].position()

                if cfg.mode == "mahalanobis":
                    if not dist_ok(p, pos, cfg, inv_cov=inv_cov):
                        continue
                    dvec = p - pos
                    dval = float(np.sqrt(dvec.T @ inv_cov @ dvec))
                    if best_d is None or dval < best_d:
                        best_d = dval
                        best_i = i_agg
                else:
                    if not dist_ok(p, pos, cfg):
                        continue
                    dval = float(np.linalg.norm(p - pos))
                    if best_d is None or dval < best_d:
                        best_d = dval
                        best_i = i_agg

            if best_i is None:
                agg = AggNode(
                    id=-1,
                    type=typ,
                    name=node.get("name", "unknown"),
                    names=list(node["names"]),
                    pos_sum=pos.copy(),
                    pos_count=1,
                    merged_from=[int(node.get("id", -1))],
                )
                aggs.append(agg)
                add_to_grid(len(aggs)-1)
            else:
                aggs[best_i].names = union_names(aggs[best_i].names, node["names"])
                aggs[best_i].pos_sum += pos
                aggs[best_i].pos_count += 1
                aggs[best_i].merged_from.append(int(node.get("id", -1)))
                # grid not updated for moved centroid (cheap approx)

        for a in aggs:
            a.id = next_id
            next_id += 1
            pos_final = a.position()
            names_final = sorted(set(a.names))
            nout = {
                "id": a.id,
                "type": a.type,
                "names": names_final,
                "name": canonical_name(names_final, a.name),
                "position": [float(pos_final[0]), float(pos_final[1]), float(pos_final[2])],
            }
            if keep_debug_fields:
                nout["_merged_from"] = a.merged_from
                nout["_count"] = a.pos_count
            out_nodes.append(nout)

    return out_nodes


# -------------------------
# CLI
# -------------------------

def parse_thr_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, type=str)
    ap.add_argument("--out_json", default="", type=str)

    ap.add_argument("--mode", default="l2",
                    choices=["l2", "l2_xy", "weighted_l2", "mahalanobis", "axis", "hybrid"])
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for l2/l2_xy/weighted_l2/mahalanobis")
    ap.add_argument("--grid_cell", type=float, default=0.0, help="0 = auto, else explicit cell size")

    ap.add_argument("--scale", type=str, default="1,1,1", help="for weighted_l2: sx,sy,sz (larger => less sensitive)")

    ap.add_argument("--axis_thr", type=str, default="0.3,0.3,0.3", help="for axis: tx,ty,tz")
    ap.add_argument("--axis_also_l2", type=float, default=-1.0, help="for axis: also require l2 < this (optional)")

    ap.add_argument("--thr_xy", type=float, default=0.35, help="for hybrid")
    ap.add_argument("--thr_z", type=float, default=0.2, help="for hybrid")

    ap.add_argument("--maha_eps", type=float, default=1e-6)

    ap.add_argument("--scan", action="store_true", help="scan multiple thresholds (thr_list)")
    ap.add_argument("--thr_list", type=str, default="", help="comma list, e.g. 0.2,0.3,0.4")
    ap.add_argument("--keep_debug", action="store_true")

    args = ap.parse_args()

    inp = Path(args.in_json)
    data = load_json(inp)
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")

    cfg = DistConfig(
        mode=args.mode,
        thr=float(args.thr),
        scale=tuple(float(x) for x in args.scale.split(",")),
        axis_thr=tuple(float(x) for x in args.axis_thr.split(",")),
        axis_also_l2=(None if args.axis_also_l2 < 0 else float(args.axis_also_l2)),
        thr_xy=float(args.thr_xy),
        thr_z=float(args.thr_z),
        maha_eps=float(args.maha_eps),
    )
    grid_cell = None if args.grid_cell <= 0 else float(args.grid_cell)

    def run_once(cfg_local: DistConfig) -> Tuple[dict, List[dict]]:
        merged = rollback_merge_nodes(nodes, cfg_local, grid_cell=grid_cell, keep_debug_fields=args.keep_debug)

        out = dict(data)
        out["nodes"] = merged

        st = dict(out.get("stats", {}))
        st["nodes_in"] = len(nodes)
        st["nodes_out"] = len(merged)
        out["stats"] = st

        mp = dict(out.get("merge_params", {}))
        mp["post_merge_mode"] = cfg_local.mode
        if cfg_local.mode in ("l2", "l2_xy", "weighted_l2", "mahalanobis"):
            mp["post_merge_thr"] = cfg_local.thr
        if cfg_local.mode == "weighted_l2":
            mp["post_merge_scale"] = list(cfg_local.scale)
        if cfg_local.mode == "axis":
            mp["post_merge_axis_thr"] = list(cfg_local.axis_thr)
            mp["post_merge_axis_also_l2"] = cfg_local.axis_also_l2
        if cfg_local.mode == "hybrid":
            mp["post_merge_thr_xy"] = cfg_local.thr_xy
            mp["post_merge_thr_z"] = cfg_local.thr_z
        if cfg_local.mode == "mahalanobis":
            mp["post_merge_maha_eps"] = cfg_local.maha_eps
        mp["post_merge_name_gate"] = "same name OR names overlap"
        out["merge_params"] = mp

        return out, merged

    if args.scan:
        thr_list = parse_thr_list(args.thr_list) if args.thr_list else [args.thr]
        print(f"[SCAN] mode={cfg.mode}, grid_cell={'auto' if grid_cell is None else grid_cell}")
        for t in thr_list:
            cfg2 = DistConfig(**{**cfg.__dict__})
            cfg2.thr = float(t)
            out_obj, merged = run_once(cfg2)
            n_out = len(merged)
            c = {}
            for n in merged:
                c[n["type"]] = c.get(n["type"], 0) + 1
            print(f"  thr={t:.4f} -> nodes_out={n_out} (objects={c.get('object',0)}, affords={c.get('affordance',0)})")
        return

    out_obj, _ = run_once(cfg)

    if args.out_json:
        outp = Path(args.out_json)
    else:
        outp = inp.with_suffix(".postmerge.json")

    save_json(outp, out_obj)
    print(f"[OK] wrote: {outp}  nodes_in={len(nodes)} -> nodes_out={len(out_obj['nodes'])}")


if __name__ == "__main__":
    main()
