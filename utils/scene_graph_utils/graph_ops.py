# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import VoxelCloud, voxel_iou
from .io import normalize_name


# -------------------------
# Semantic group
# -------------------------

@dataclass
class SemanticGroup:
    """
    Semantic group that accumulates WORLD-space points for (type, name, inst_id).
    """
    typ: str
    name: str  # canonical name
    inst_id: str
    cloud: VoxelCloud
    raw_names: set = field(default_factory=set)
    observations: int = field(default=0)
    sample_masks: List[str] = field(default_factory=list)
    # indices into the original scene point cloud (world_pts)
    point_indices: set = field(default_factory=set)

    def add_obs(
        self,
        raw_name: str,
        mask_name: str,
        pts_world: np.ndarray,
        pt_indices: Optional[np.ndarray] = None,
        max_sample_masks: int = 10,
    ):
        self.raw_names.add(normalize_name(raw_name))
        self.observations += 1
        if mask_name and len(self.sample_masks) < max_sample_masks:
            self.sample_masks.append(mask_name)
        self.cloud.add(pts_world)
        if pt_indices is not None:
            # store as Python ints for JSON-serializable set later
            for idx in np.asarray(pt_indices, dtype=np.int64).ravel():
                self.point_indices.add(int(idx))

    def all_names(self) -> List[str]:
        s = set(self.raw_names)
        s.add(self.name)
        return sorted({normalize_name(x) for x in s if x})


# -------------------------
# Merging (same type+name; ignore inst_id) with distance + voxel IoU
# -------------------------

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def compute_merge_center(g: SemanticGroup) -> Optional[np.ndarray]:
    """
    Default merge center:
      1) DBSCAN largest-cluster centroid
      2) fallback to voxel-cloud centroid if DBSCAN fails
    """
    pts = g.cloud.points()
    if pts.shape[0] == 0:
        return None

    # ---- DBSCAN (hard-coded defaults, no CLI) ----
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        labels = np.array(
            pcd.cluster_dbscan(
                eps=0.10,          # keep consistent with dbscan_eps default
                min_points=10,     # keep consistent with dbscan_min_samples default
                print_progress=False,
            ),
            dtype=np.int32,
        )

        valid = labels >= 0
        if np.any(valid):
            ids, counts = np.unique(labels[valid], return_counts=True)
            best_id = ids[np.argmax(counts)]
            c = pts[labels == best_id].mean(axis=0)
            if np.isfinite(c).all():
                return c.astype(np.float32)
    except Exception:
        pass

    # ---- fallback ----
    c = g.cloud.centroid()
    if c is None or (not np.isfinite(c).all()):
        return None
    return c.astype(np.float32)


def merge_same_name_groups(
    groups: List[SemanticGroup],
    eps: float,
    voxel_iou_min: float,
) -> Tuple[List[SemanticGroup], List[List[int]]]:
    """
    For each bucket of (type, name), merge groups across different inst_id if:
      - merge-center distance < eps
      - voxel IoU >= voxel_iou_min

    NOTE:
      - merge center is computed by compute_merge_center()
      - Union-Find single-linkage is kept (chaining still possible)
    """
    if not groups:
        return [], []

    # bucket by (type, name)
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for i, g in enumerate(groups):
        buckets.setdefault((g.typ, g.name), []).append(i)

    merged: List[SemanticGroup] = []
    comps_out: List[List[int]] = []

    for (typ, name), idxs in buckets.items():
        if len(idxs) == 1:
            merged.append(groups[idxs[0]])
            comps_out.append([idxs[0]])
            continue

        L = len(idxs)
        uf = UnionFind(L)

        # ---- compute merge centers ONCE (cache) ----
        centers: List[Optional[np.ndarray]] = []
        for gi in idxs:
            c = compute_merge_center(groups[gi])
            centers.append(c)

        # ---- pairwise union ----
        for a in range(L):
            ca = centers[a]
            if ca is None:
                continue

            for b in range(a + 1, L):
                cb = centers[b]
                if cb is None:
                    continue

                dist = float(np.linalg.norm(ca - cb))
                if dist >= eps:
                    continue

                iou = voxel_iou(
                    groups[idxs[a]].cloud,
                    groups[idxs[b]].cloud,
                )
                if iou >= voxel_iou_min:
                    uf.union(a, b)

        # ---- collect components ----
        comp: Dict[int, List[int]] = {}
        for a in range(L):
            r = uf.find(a)
            comp.setdefault(r, []).append(a)

        # ---- build merged SemanticGroup per component ----
        for _, local_ids in comp.items():
            src = [idxs[k] for k in local_ids]

            def score(gi: int):
                return (groups[gi].observations, groups[gi].cloud.size())

            rep = max(src, key=score)
            rep_g = groups[rep]

            out_cloud = VoxelCloud(rep_g.cloud.v)
            out_point_indices: set = set()
            raw_names = set()
            obs = 0
            sm: List[str] = []
            seen = set()

            for gi in src:
                g_src = groups[gi]
                out_cloud.merge_from(g_src.cloud)
                raw_names |= set(g_src.raw_names)
                obs += int(g_src.observations)
            if hasattr(g_src, "point_indices"):
                out_point_indices |= set(g_src.point_indices)

                for x in g_src.sample_masks:
                    if x not in seen:
                        seen.add(x)
                        sm.append(x)
                    if len(sm) >= 10:
                        break
                if len(sm) >= 10:
                    break

            out = SemanticGroup(
                typ=typ,
                name=name,
                inst_id=rep_g.inst_id,
                cloud=out_cloud,
                raw_names=raw_names,
                observations=obs,
                sample_masks=sm,
                point_indices=out_point_indices,
            )

            merged.append(out)
            comps_out.append(src)

    return merged, comps_out


# -------------------------
# InstReuseGuard (DIST ONLY)
# -------------------------

def make_branch_inst_id(base_inst_id: str, typ: str, cname: str, groups_map: Dict[Tuple[str, str, str], SemanticGroup]) -> str:
    """
    Create a new inst_id branch like "000__b01" that does not collide with existing keys.
    """
    n = 1
    while True:
        cand = f"{base_inst_id}__b{n:02d}"
        if (typ, cname, cand) not in groups_map:
            return cand
        n += 1


def resolve_inst_id_with_guard(
    typ: str,
    cname: str,
    base_inst_id: str,
    pts_new: np.ndarray,
    groups_map: Dict[Tuple[str, str, str], SemanticGroup],
    voxel_size: float,
    eps_split: float,
) -> Tuple[str, bool, float]:
    """
    Decide which inst_id to use under (typ,cname,base_inst_id) when inst_id may be reused.

    Strategy:
      - consider existing candidates: base_inst_id and all branches base_inst_id__bXX
      - compute centroid distance to each candidate (using voxel centroid)
      - if nearest dist <= eps_split -> reuse that candidate inst_id
      - else create a new branch inst_id and return it

    Returns:
      chosen_inst_id, created_new_branch, nearest_dist
    """
    # Build candidates: base + branches
    prefix = f"{base_inst_id}__b"
    candidates: List[Tuple[str, SemanticGroup]] = []

    k0 = (typ, cname, base_inst_id)
    if k0 in groups_map:
        candidates.append((base_inst_id, groups_map[k0]))

    for (t, n, iid), g in groups_map.items():
        if t == typ and n == cname and iid.startswith(prefix):
            candidates.append((iid, g))

    # If nothing exists yet -> use base inst id
    if not candidates:
        return base_inst_id, False, 0.0

    # Build new cloud centroid
    tmp = VoxelCloud(voxel_size=float(voxel_size))
    tmp.add(pts_new)
    c_new = tmp.centroid()
    if c_new is None or tmp.size() == 0:
        # degenerate, don't branch
        return base_inst_id, False, 0.0

    # Find nearest existing candidate
    best_inst = candidates[0][0]
    best_dist = float("inf")

    for iid, g in candidates:
        c_old = g.cloud.centroid()
        if c_old is None or g.cloud.size() == 0:
            continue
        d = float(np.linalg.norm(c_old.astype(np.float64) - c_new.astype(np.float64)))
        if d < best_dist:
            best_dist = d
            best_inst = iid

    # If close enough -> reuse nearest
    if best_dist <= float(eps_split):
        return best_inst, False, best_dist

    # Else create new branch
    new_inst = make_branch_inst_id(base_inst_id, typ, cname, groups_map)
    return new_inst, True, best_dist


# -------------------------
# Affordance assignment
# -------------------------

def assign_affordances_to_nearest_object(nodes: List[dict]) -> List[dict]:
    objects = [n for n in nodes if n.get("type") == "object"]
    affords = [n for n in nodes if n.get("type") == "affordance"]

    if not affords:
        return nodes

    if not objects:
        for a in affords:
            a["belong_to"] = None
        return nodes

    obj_pos = np.array([o["position"] for o in objects], dtype=np.float64)
    obj_ids = [o["id"] for o in objects]

    for a in affords:
        ap = np.array(a["position"], dtype=np.float64)
        d = np.linalg.norm(obj_pos - ap.reshape(1, 3), axis=1)
        a["belong_to"] = int(obj_ids[int(np.argmin(d))])

    return nodes
