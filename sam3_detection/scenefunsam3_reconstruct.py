"""
python scenefunsam3_reconstruct.py \
  --masks_root /nas/qirui/sam3/scenefun3d_ex/420683/contextual_obj_masks \
  --depth_dir /nas/qirui/scenefun3d/data/420683/42445137/hires_depth \
  --depth_glob "*.png" \
  --traj /nas/qirui/scenefun3d/data/420683/42445137/hires_poses.traj \
  --intr_dir /nas/qirui/scenefun3d/data/420683/42445137/hires_wide_intrinsics \
  --out_dir /nas/qirui/sam3/scenefun3d_ex/420683/pcd \
  --depth_scale 1000 \
  --backproj_stride 1 \
  --cov_debug --cov_only_first_instance \
  --assoc_enable \
  --assoc_log_every 1 \
  --voxel_hash 0.01 \
  --eps_depth 0.05 \
  --tau_occ 0.05 \
  --support_keep 3 \
  --ratio_keep 0.6 \
  --reject_drop 4 \
  --stale_frames 20 \
  --prune_every 1 \
  --assoc_only


"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

try:
    import cv2
except Exception:
    cv2 = None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_frame_indices(masks_root: Path) -> List[int]:
    subdirs = [p for p in masks_root.iterdir() if p.is_dir()]
    numeric = []
    for p in subdirs:
        if re.fullmatch(r"\d+", p.name):
            numeric.append(int(p.name))
    if numeric:
        return sorted(numeric)
    return [0]

def frame_mask_dir(masks_root: Path, idx: int) -> Path:
    p = masks_root / str(idx)
    if p.is_dir():
        return p
    return masks_root

def read_lines(p: Path) -> List[str]:
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def read_traj(traj_path: Path) -> List[np.ndarray]:
    """
    Reads a .traj with 7 fields per line:
        <id_or_ts> ax ay az tx ty tz
    where (ax,ay,az) is Rodrigues angle-axis, and (tx,ty,tz) translation.
    Returns a list of 4x4 transforms, in the SAME convention as the traj file lines.
    In main(), if --traj_is_world_to_cam is set, we invert to get T_wc.
    """
    T_list = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(f"Bad traj line (expect 7 fields): {line}")
            _, ax, ay, az, tx, ty, tz = map(float, parts)
            rvec = np.array([ax, ay, az], dtype=np.float64)
            R = rodrigues(rvec)
            t = np.array([tx, ty, tz], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            T_list.append(T)
    if not T_list:
        raise ValueError(f"No valid poses in traj: {traj_path}")
    return T_list


def read_intrinsics_file(p: Path) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Expects: width height fx fy cx cy
    """
    txt = p.read_text(encoding="utf-8").strip().split()
    if len(txt) != 6:
        raise ValueError(f"Bad intrinsics file {p}, expect: width height fx fy cx cy")
    w, h = int(float(txt[0])), int(float(txt[1]))
    fx, fy, cx, cy = map(float, txt[2:])
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    """
    Loads depth as float32 in meters (or unit implied by depth_scale).
    - If .npy: assumes already numeric.
    - Else: reads image then divides by depth_scale.
    """
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth: {depth_path}")
    if depth_path.suffix.lower() == ".npy":
        d = np.load(depth_path).astype(np.float32)
    else:
        if cv2 is None:
            img = o3d.io.read_image(str(depth_path))
            d = np.asarray(img).astype(np.float32)
        else:
            d_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d_raw is None:
                raise ValueError(f"Failed to read depth: {depth_path}")
            d = d_raw.astype(np.float32)

    d_m = d / float(depth_scale)
    d_m[~np.isfinite(d_m)] = 0.0
    d_m[d_m < 0] = 0.0
    return d_m.astype(np.float32)


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Loads a mask, returns uint8 {0,1}. NO cleanup in this version.
    """
    if cv2 is not None:
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        if m.ndim == 3:
            m = m[..., 0]
    else:
        img = o3d.io.read_image(str(mask_path))
        m = np.asarray(img)
        if m.ndim == 3:
            m = m[..., 0]
    return (m > 0).astype(np.uint8)


def backproject_masked_depth(depth_m: np.ndarray,
                             mask01: np.ndarray,
                             intr: o3d.camera.PinholeCameraIntrinsic,
                             depth_trunc: float,
                             stride: int) -> np.ndarray:
    """
    Backproject masked depth to camera coordinates.
    Returns Nx3 points in camera frame.
    """
    if depth_m.shape[:2] != mask01.shape[:2]:
        raise ValueError(f"Depth/mask shape mismatch: depth={depth_m.shape} mask={mask01.shape}")

    h, w = depth_m.shape[:2]
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    m = mask01[yv, xv] > 0
    z = depth_m[yv, xv]
    valid = m & (z > 0) & (z < depth_trunc)

    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xv - cx) * z / fx
    y = (yv - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def make_o3d_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if pts.size:
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transform to Nx3 points.
    """
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def project_zbuffer(pts_cam: np.ndarray,
                    intr: o3d.camera.PinholeCameraIntrinsic,
                    width: int,
                    height: int) -> np.ndarray:
    """
    Vectorized z-buffer projection.
    """
    if pts_cam.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    if z.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    u = np.round(x * fx / z + cx).astype(np.int32)
    v = np.round(y * fy / z + cy).astype(np.int32)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    if z.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    order = np.argsort(z)[::-1]  # far->near so near overwrites
    u_sorted = u[order]
    v_sorted = v[order]
    z_sorted = z[order]

    buf = np.zeros((height, width), dtype=np.float32)
    buf[v_sorted, u_sorted] = z_sorted
    return buf


def bbox_from_valid(valid_mask: np.ndarray):
    ys, xs = np.where(valid_mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def per_frame_self_coverage(depth_m, mask01, intr, T_wc_eff, gp, W, H):
    """
    STRICT self-coverage (NO cross-frame):
      observed: depth ⊙ mask on THIS frame
      expected: project gp (global_pcd AFTER integrate for THIS frame) back to THIS frame
      coverage = |obs_valid & exp_valid| / |obs_valid|
    """
    if gp is None or len(gp.points) == 0:
        return 0.0, 0, 0, 0, None, None

    observed = depth_m.copy()
    observed[mask01 == 0] = 0.0
    obs_valid = observed > 0

    pts_w = np.asarray(gp.points)
    T_cw = np.linalg.inv(T_wc_eff)
    pts_cam = (T_cw[:3, :3] @ pts_w.T + T_cw[:3, 3:4]).T.astype(np.float32)

    expected = project_zbuffer(pts_cam, intr, W, H)
    exp_valid = (expected > 0).astype(np.uint8)

    # optional mild dilation for sparse projection
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        exp_valid = cv2.dilate(exp_valid, kernel, iterations=1)

    exp_valid_bool = exp_valid > 0
    in_exp = obs_valid & exp_valid_bool
    cov = float(in_exp.sum()) / float(max(1, obs_valid.sum()))

    return cov, int(obs_valid.sum()), int(exp_valid.sum()), int(in_exp.sum()), bbox_from_valid(obs_valid), bbox_from_valid(exp_valid)


# ----------------------------
# Association point pool (ADD + DELETE)
# ----------------------------

@dataclass
class PoolPoint:
    xyz: np.ndarray
    support: int = 1
    reject: int = 0
    last_seen: int = 0


class PointPool:
    """
    Voxel-hash dedup point pool.
    One representative point per voxel cell.
    """
    def __init__(self, voxel_hash: float):
        self.voxel = float(voxel_hash)
        self.keys: Dict[Tuple[int, int, int], int] = {}
        self.points: List[PoolPoint] = []

    def _key(self, p: np.ndarray) -> Tuple[int, int, int]:
        v = self.voxel
        return (int(np.floor(p[0] / v)),
                int(np.floor(p[1] / v)),
                int(np.floor(p[2] / v)))

    def add_points(self, pts_w: np.ndarray, frame_idx: int) -> int:
        added = 0
        for p in pts_w:
            k = self._key(p)
            if k in self.keys:
                continue
            self.keys[k] = len(self.points)
            self.points.append(PoolPoint(xyz=p.astype(np.float32), support=1, reject=0, last_seen=frame_idx))
            added += 1
        return added

    def prune(self,
              frame_idx: int,
              support_keep: int,
              ratio_keep: float,
              reject_drop: int,
              stale_frames: int) -> int:
        """
        Hysteresis prune:
          keep if support>=support_keep OR support_ratio>=ratio_keep
          drop if reject>=reject_drop AND stale>=stale_frames AND not must_keep
        """
        if not self.points:
            return 0

        keep_mask = np.ones(len(self.points), dtype=bool)
        for i, pp in enumerate(self.points):
            denom = max(1, pp.support + pp.reject)
            ratio = pp.support / float(denom)
            must_keep = (pp.support >= support_keep) or (ratio >= ratio_keep)
            must_drop = (pp.reject >= reject_drop) and ((frame_idx - pp.last_seen) >= stale_frames)
            if must_drop and (not must_keep):
                keep_mask[i] = False

        if keep_mask.all():
            return 0

        removed = int((~keep_mask).sum())

        new_points: List[PoolPoint] = []
        new_keys: Dict[Tuple[int, int, int], int] = {}
        for old_i, ok in enumerate(keep_mask):
            if not ok:
                continue
            pp = self.points[old_i]
            k = self._key(pp.xyz)
            new_keys[k] = len(new_points)
            new_points.append(pp)

        self.points = new_points
        self.keys = new_keys
        return removed


def associate_update_pool(pool: PointPool,
                          intr: o3d.camera.PinholeCameraIntrinsic,
                          depth_m: np.ndarray,
                          mask01: np.ndarray,
                          T_wc: np.ndarray,
                          eps_depth: float,
                          tau_occ: float,
                          depth_trunc: float,
                          frame_idx: int) -> Dict[str, int]:
    """
    For each pool point:
      - reproject into current frame
      - support if (in mask) AND (|d - z_pred| < eps_depth)
      - reject if visible and not occluded but fails support
      - skip if out-of-FOV/invalid depth/occluded/behind
    """
    if not pool.points:
        return dict(eval=0, support=0, reject=0, skip=0)

    H, W = depth_m.shape[:2]
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    T_cw = np.linalg.inv(T_wc)

    n_eval = n_sup = n_rej = n_skip = 0

    for pp in pool.points:
        # world -> cam
        Xc, Yc, Zc = transform_points(T_cw, pp.xyz.reshape(1, 3)).reshape(3)
        if Zc <= 0 or Zc >= depth_trunc:
            n_skip += 1
            continue

        u = int(np.round(Xc * fx / Zc + cx))
        v = int(np.round(Yc * fy / Zc + cy))
        if u < 0 or u >= W or v < 0 or v >= H:
            n_skip += 1
            continue

        d = float(depth_m[v, u])
        if d <= 0 or d >= depth_trunc:
            n_skip += 1
            continue

        # occlusion skip: observed surface is much closer than predicted
        if d < (Zc - tau_occ):
            n_skip += 1
            continue

        n_eval += 1

        in_mask = (mask01[v, u] > 0)
        depth_ok = (abs(d - Zc) < eps_depth)

        if in_mask and depth_ok:
            pp.support += 1
            pp.last_seen = frame_idx
            n_sup += 1
        else:
            pp.reject += 1
            n_rej += 1

    return dict(eval=n_eval, support=n_sup, reject=n_rej, skip=n_skip)


def pool_to_pcd(pool: PointPool,
                color_by_ratio: bool,
                fixed_color: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if not pool.points:
        return pcd

    pts = np.stack([pp.xyz for pp in pool.points], axis=0).astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)

    if fixed_color is not None:
        col = np.tile(fixed_color.reshape(1, 3), (len(pool.points), 1)).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(col)
        return pcd

    if color_by_ratio:
        cols = np.zeros((len(pool.points), 3), dtype=np.float64)
        for i, pp in enumerate(pool.points):
            denom = max(1, pp.support + pp.reject)
            r = pp.support / float(denom)  # 0..1
            cols[i] = np.array([r, 0.2 + 0.8 * (1.0 - abs(r - 0.5) * 2.0), 1.0 - r], dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


# ----------------------------
# TSDF reconstructor per instance
# ----------------------------

class InstanceReconstructor:
    def __init__(self, voxel_length: float, sdf_trunc: float, depth_trunc: float):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_length),
            sdf_trunc=float(sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )
        self.depth_trunc = float(depth_trunc)
        self.global_pcd = o3d.geometry.PointCloud()

    def integrate_masked_depth(self,
                               depth_m: np.ndarray,
                               mask01: np.ndarray,
                               intr: o3d.camera.PinholeCameraIntrinsic,
                               T_wc: np.ndarray) -> None:
        """
        Open3D TSDF integrate() expects EXTRINSIC = world->camera (T_cw).
        We keep poses as T_wc, so we invert here.
        """
        h, w = depth_m.shape[:2]
        depth_masked = depth_m.copy()
        depth_masked[mask01 == 0] = 0.0

        depth_o3d = o3d.geometry.Image((depth_masked * 1000.0).astype(np.uint16))
        color_o3d = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )
        T_cw = np.linalg.inv(T_wc)
        self.volume.integrate(rgbd, intr, T_cw)

    def update_global_pcd(self, voxel_down: float) -> o3d.geometry.PointCloud:
        pcd = self.volume.extract_point_cloud()
        if voxel_down > 0 and len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_down)
        self.global_pcd = pcd
        return pcd


# ----------------------------
# main
# ----------------------------

def build_depth_index(depth_dir: Path,
                      depth_glob: str,
                      depth_list: Optional[Path]) -> List[Path]:
    if depth_list is not None:
        names = read_lines(depth_list)
        files = [depth_dir / n for n in names]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Some depth paths in depth_list do not exist, e.g. {missing[:5]}")
        return files

    files = list(depth_dir.glob(depth_glob))
    if not files:
        raise ValueError(f"No depth files found in {depth_dir} with glob '{depth_glob}'")
    return sorted(files, key=lambda p: natural_key(p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_root", type=str, required=True)
    ap.add_argument("--depth_dir", type=str, required=True)
    ap.add_argument("--depth_glob", type=str, default="*.png")
    ap.add_argument("--depth_list", type=str, default="")
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_trunc", type=float, default=5.0)

    ap.add_argument("--traj", type=str, required=True)
    ap.add_argument("--traj_is_world_to_cam", action="store_true",
                    help="Set if traj pose is world-to-camera (T_cw). Will invert to get T_wc.")

    ap.add_argument("--intr_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # TSDF params (original)
    ap.add_argument("--voxel_length", type=float, default=0.01)
    ap.add_argument("--sdf_trunc", type=float, default=0.04)
    ap.add_argument("--pcd_voxel_down", type=float, default=0.01)

    ap.add_argument("--backproj_stride", type=int, default=1)

    ap.add_argument("--skip_empty", action="store_true")

    # ---- coverage debug controls (TSDF)
    ap.add_argument("--cov_debug", action="store_true")
    ap.add_argument("--cov_only_first_instance", action="store_true")
    ap.add_argument("--cov_every", type=int, default=1)

    ap.add_argument("--print_intr_01", action="store_true")

    # ---- first-frame-only backprojection mode ----
    ap.add_argument("--only_first_frame_backproj", action="store_true")
    ap.add_argument("--first_frame_idx", type=int, default=0)
    ap.add_argument("--backproj_out_stride", type=int, default=1)

    # ----------------------------
    # NEW: association (add/delete) switches + params
    # ----------------------------
    ap.add_argument("--assoc_enable", action="store_true",
                    help="Enable dynamic add/delete association point pool.")
    ap.add_argument("--assoc_only", action="store_true",
                    help="If set, skip TSDF integrate and ONLY run association point pool.")
    ap.add_argument("--assoc_log_every", type=int, default=1,
                    help="Print assoc stats every N frames.")
    ap.add_argument("--voxel_hash", type=float, default=0.01,
                    help="Voxel size for point-pool dedup (meters).")
    ap.add_argument("--eps_depth", type=float, default=0.05,
                    help="Depth consistency threshold |d - z_pred| < eps (meters).")
    ap.add_argument("--tau_occ", type=float, default=0.05,
                    help="Occlusion tolerance: if d < z_pred - tau_occ, skip (meters).")
    ap.add_argument("--support_keep", type=int, default=3)
    ap.add_argument("--ratio_keep", type=float, default=0.6)
    ap.add_argument("--reject_drop", type=int, default=4)
    ap.add_argument("--stale_frames", type=int, default=20)
    ap.add_argument("--prune_every", type=int, default=1)
    ap.add_argument("--assoc_color_by_ratio", action="store_true",
                    help="Color association output by support/(support+reject).")

    args = ap.parse_args()

    masks_root = Path(args.masks_root)
    depth_dir = Path(args.depth_dir)
    intr_dir = Path(args.intr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_list_path = Path(args.depth_list) if args.depth_list.strip() else None
    depth_files = build_depth_index(depth_dir, args.depth_glob, depth_list_path)

    T_list = read_traj(Path(args.traj))

    intr_files = sorted([p for p in intr_dir.iterdir() if p.is_file()], key=lambda p: natural_key(p.name))
    if not intr_files:
        raise ValueError(f"No intrinsics files found in {intr_dir}")

    frame_ids = list_frame_indices(masks_root)
    max_idx = max(frame_ids)

    # full pipeline sanity checks (skip these when only_first_frame_backproj)
    if not args.only_first_frame_backproj:
        if max_idx >= len(depth_files):
            raise ValueError(f"Depth files count={len(depth_files)} but need at least {max_idx+1}. "
                             f"Fix by using --depth_list or adjust --depth_glob.")
        if max_idx >= len(T_list):
            raise ValueError(f"Traj poses count={len(T_list)} but need at least {max_idx+1}.")
        if max_idx >= len(intr_files):
            raise ValueError(f"Intrinsics files count={len(intr_files)} but need at least {max_idx+1}.")

    if args.print_intr_01 and len(intr_files) >= 2:
        intr0 = read_intrinsics_file(intr_files[0])
        intr1 = read_intrinsics_file(intr_files[1])
        print("intr0:", intr0.intrinsic_matrix)
        print("intr1:", intr1.intrinsic_matrix)

    # ----------------------------
    # ONLY first-frame backprojection mode (unchanged)
    # ----------------------------
    if args.only_first_frame_backproj:
        idx = int(args.first_frame_idx)

        if idx >= len(depth_files):
            raise ValueError(f"first_frame_idx={idx} but depth_files has only {len(depth_files)} entries")
        if idx >= len(T_list):
            raise ValueError(f"first_frame_idx={idx} but traj has only {len(T_list)} poses")
        if idx >= len(intr_files):
            raise ValueError(f"first_frame_idx={idx} but intr_files has only {len(intr_files)} entries")

        intr = read_intrinsics_file(intr_files[idx])
        W, H = intr.width, intr.height

        T = T_list[idx].copy()
        if args.traj_is_world_to_cam:
            T = np.linalg.inv(T)
        T_wc = T

        depth_path = depth_files[idx]
        depth_m = load_depth(depth_path, args.depth_scale)
        if depth_m.shape[0] != H or depth_m.shape[1] != W:
            raise ValueError(f"[frame {idx}] depth size {depth_m.shape[1]}x{depth_m.shape[0]} != intr {W}x{H}. depth={depth_path.name}")

        frame_dir = frame_mask_dir(masks_root, idx)

        mask_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
                            key=lambda p: natural_key(p.name))
        if not mask_paths:
            raise ValueError(f"No mask images found in {frame_dir} for first-frame-only mode")

        instance_names_ff = sorted({p.stem for p in mask_paths}, key=natural_key)
        print(f"[FF] frame={idx:04d} instances({len(instance_names_ff)}): {instance_names_ff}")

        rng = np.random.default_rng(0)
        inst_colors_ff = {name: rng.random(3) for name in instance_names_ff}

        merged = o3d.geometry.PointCloud()

        for name in instance_names_ff:
            mp = None
            for suf in [".png", ".jpg", ".jpeg"]:
                p2 = frame_dir / f"{name}{suf}"
                if p2.exists():
                    mp = p2
                    break
            if mp is None:
                continue

            mask01 = load_mask(mp)
            if args.skip_empty and mask01.sum() == 0:
                continue

            pts_cam = backproject_masked_depth(
                depth_m, mask01, intr,
                depth_trunc=args.depth_trunc,
                stride=max(1, int(args.backproj_out_stride))
            )
            if pts_cam.shape[0] == 0:
                print(f"[FF] no valid depth points: {name}")
                continue

            pts_w = transform_points(T_wc, pts_cam)
            pcd = make_o3d_pcd(pts_w)

            col = np.tile(inst_colors_ff[name][None, :], (len(pcd.points), 1)).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(col)

            out_ply = out_dir / f"FF_frame{idx:04d}_{name}_backproj.ply"
            o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False, compressed=False)
            print(f"[FF-SAVE] {out_ply} points={len(pcd.points)}")

            merged += pcd

        merged_ply = out_dir / f"FF_frame{idx:04d}_all_instances_backproj_merged.ply"
        o3d.io.write_point_cloud(str(merged_ply), merged, write_ascii=False, compressed=False)
        print(f"[FF-SAVE] {merged_ply} points={len(merged.points)}")

        try:
            o3d.visualization.draw_geometries([merged], window_name=f"FF backproj frame {idx:04d}")
        except Exception as e:
            print(f"[FF] visualization skipped: {e}")

        print("[FF-DONE]")
        return

    # ----------------------------
    # Full pipeline: discover instance names
    # ----------------------------
    first_dir = frame_mask_dir(masks_root, frame_ids[0])
    mask_paths0 = sorted([p for p in first_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
                         key=lambda p: natural_key(p.name))
    if not mask_paths0:
        raise ValueError(f"No mask images found in {first_dir}")

    instance_names = sorted({p.stem for p in mask_paths0}, key=natural_key)
    print(f"[INFO] instances ({len(instance_names)}): {instance_names}")

    rng = np.random.default_rng(0)
    inst_colors = {name: rng.random(3) for name in instance_names}

    # TSDF recon structures (optional)
    recon: Dict[str, InstanceReconstructor] = {}
    if (not args.assoc_only):
        recon = {
            name: InstanceReconstructor(args.voxel_length, args.sdf_trunc, args.depth_trunc)
            for name in instance_names
        }

    # Association point pools (optional)
    pools: Dict[str, PointPool] = {}
    if args.assoc_enable:
        pools = {name: PointPool(args.voxel_hash) for name in instance_names}

    cov_log = []  # TSDF coverage log

    # ----------------------------
    # Process frames
    # ----------------------------
    for idx in frame_ids:
        intr = read_intrinsics_file(intr_files[idx])
        W, H = intr.width, intr.height

        T = T_list[idx].copy()
        if args.traj_is_world_to_cam:
            T = np.linalg.inv(T)
        T_wc = T
        T_wc_eff = T_wc  # no ICP

        depth_path = depth_files[idx]
        depth_m = load_depth(depth_path, args.depth_scale)
        if depth_m.shape[0] != H or depth_m.shape[1] != W:
            raise ValueError(f"[frame {idx}] depth size {depth_m.shape[1]}x{depth_m.shape[0]} != intr {W}x{H}. depth={depth_path.name}")

        frame_dir = frame_mask_dir(masks_root, idx)

        for name in instance_names:
            mp = frame_dir / f"{name}.png"
            if not mp.exists():
                continue

            mask01 = load_mask(mp)
            if args.skip_empty and mask01.sum() == 0:
                continue

            # ---- Association: ADD + ASSOC + PRUNE ----
            if args.assoc_enable:
                # ADD: new points from this frame
                pts_cam = backproject_masked_depth(
                    depth_m, mask01, intr,
                    depth_trunc=args.depth_trunc,
                    stride=max(1, int(args.backproj_stride))
                )
                added = 0
                if pts_cam.shape[0] > 0:
                    pts_w = transform_points(T_wc_eff, pts_cam)
                    added = pools[name].add_points(pts_w, frame_idx=idx)

                # ASSOC: update support/reject with this frame's mask+depth
                stat = associate_update_pool(
                    pools[name], intr, depth_m, mask01, T_wc_eff,
                    eps_depth=args.eps_depth,
                    tau_occ=args.tau_occ,
                    depth_trunc=args.depth_trunc,
                    frame_idx=idx
                )

                # PRUNE: delete points with enough negative evidence
                removed = 0
                if args.prune_every > 0 and (idx % max(1, args.prune_every) == 0):
                    removed = pools[name].prune(
                        frame_idx=idx,
                        support_keep=args.support_keep,
                        ratio_keep=args.ratio_keep,
                        reject_drop=args.reject_drop,
                        stale_frames=args.stale_frames
                    )

                if (idx % max(1, args.assoc_log_every) == 0):
                    print(f"[ASSOC] frame={idx:04d} inst={name:>20s} "
                          f"add={added:>6d} pool={len(pools[name].points):>7d} rm={removed:>5d} "
                          f"(eval/sup/rej/skip)=({stat['eval']}/{stat['support']}/{stat['reject']}/{stat['skip']})")

            # ---- TSDF integrate (optional) ----
            if not args.assoc_only:
                recon[name].integrate_masked_depth(depth_m, mask01, intr, T_wc_eff)
                recon[name].update_global_pcd(0.0)

                if args.cov_debug and (idx % max(1, args.cov_every) == 0):
                    if (not args.cov_only_first_instance) or (name == instance_names[0]):
                        gp_now = recon[name].global_pcd
                        cov, obs_n, exp_n, in_n, obs_bb, exp_bb = per_frame_self_coverage(
                            depth_m, mask01, intr, T_wc_eff, gp_now, W, H
                        )
                        cov_log.append((idx, name, cov, obs_n, exp_n, in_n))
                        print(f"[COV] frame={idx:04d} inst={name} cov(in_exp/obs)={cov:.4f} "
                              f"obs={obs_n} exp={exp_n} in_exp={in_n} obs_bb={obs_bb} exp_bb={exp_bb}")

                print(f"[INFO] frame {idx:04d} depth='{depth_path.name}' inst='{name}'")

    # ----------------------------
    # Save TSDF results (if used)
    # ----------------------------
    if not args.assoc_only:
        if args.cov_debug and cov_log:
            cov_vals = [c for (_, _, c, _, _, _) in cov_log]
            print("[COV-SUM] entries=", len(cov_vals),
                  "min=", float(np.min(cov_vals)),
                  "mean=", float(np.mean(cov_vals)),
                  "max=", float(np.max(cov_vals)))

        merged_tsdf = o3d.geometry.PointCloud()
        for name in instance_names:
            pcd = recon[name].update_global_pcd(args.pcd_voxel_down)
            if len(pcd.points) > 0:
                col = np.tile(inst_colors[name][None, :], (len(pcd.points), 1)).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(col)

            out_ply = out_dir / f"{name}.ply"
            o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False, compressed=False)
            print(f"[SAVE-TSDF] {out_ply} points={len(pcd.points)}")
            merged_tsdf += pcd

        merged_ply = out_dir / "all_instances_merged.ply"
        o3d.io.write_point_cloud(str(merged_ply), merged_tsdf, write_ascii=False, compressed=False)
        print(f"[SAVE-TSDF] {merged_ply} points={len(merged_tsdf.points)}")

    # ----------------------------
    # Save Association results (if enabled)
    # ----------------------------
    if args.assoc_enable:
        merged_assoc = o3d.geometry.PointCloud()
        for name in instance_names:
            fixed_color = None if args.assoc_color_by_ratio else inst_colors[name]
            pcd = pool_to_pcd(pools[name], color_by_ratio=args.assoc_color_by_ratio, fixed_color=fixed_color)

            out_ply = out_dir / f"{name}_assoc_pool.ply"
            o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False, compressed=False)
            print(f"[SAVE-ASSOC] {out_ply} points={len(pcd.points)}")
            merged_assoc += pcd

        merged_ply = out_dir / "all_instances_assoc_pool_merged.ply"
        o3d.io.write_point_cloud(str(merged_ply), merged_assoc, write_ascii=False, compressed=False)
        print(f"[SAVE-ASSOC] {merged_ply} points={len(merged_assoc.points)}")

    print("[DONE]")


if __name__ == "__main__":
    main()
