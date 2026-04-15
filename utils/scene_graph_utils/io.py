# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .geometry import rodrigues, validate_transform


# -------------------------
# Common helpers
# -------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def normalize_name(name: str) -> str:
    """Normalize entity names for better matching."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)

    synonyms = {
        "drawers": "drawer",
        "handles": "handle",
        "doors": "door",
        "cabinets": "cabinet",
    }
    for plural, singular in synonyms.items():
        if name == plural:
            return singular
    return name


# -------------------------
# Intrinsics / traj / masks
# -------------------------

def read_traj_Twc(traj_path: Path) -> List[np.ndarray]:
    """
    Read trajectory file: timestamp ax ay az tx ty tz
    Returns T_wc (world <- camera) transforms.
    """
    T_list = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(f"Line {line_no}: expect 7 fields, got {len(parts)}")

            try:
                _, ax, ay, az, tx, ty, tz = map(float, parts)
                rvec = np.array([ax, ay, az], dtype=np.float64)
                R = rodrigues(rvec)
                t = np.array([tx, ty, tz], dtype=np.float64)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t
                validate_transform(T, f"Trajectory line {line_no}")
                T_list.append(T)
            except Exception as e:
                raise ValueError(f"Line {line_no}: failed to parse - {e}")

    if not T_list:
        raise ValueError(f"No valid poses in: {traj_path}")
    return T_list


def load_intrinsics_txt(p: Path) -> Tuple[int, int, float, float, float, float]:
    """Load intrinsics: width height fx fy cx cy"""
    vals = p.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Intrinsics {p}: expect 6 values, got {len(vals)}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:])

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal length: fx={fx}, fy={fy}")

    return w, h, fx, fy, cx, cy


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask as binary (H, W) array."""
    if cv2 is None:
        raise RuntimeError("OpenCV not available. Install: pip install opencv-python")

    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    if m.ndim == 3:
        m = m[..., 0]

    return (m > 0).astype(np.uint8)


def list_frame_dirs(masks_root: Path) -> List[int]:
    """List frame indices from mask directory structure."""
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")

    sub = [p for p in masks_root.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if not sub:
        return [0]
    return sorted([int(p.name) for p in sub])


def get_frame_dir(masks_root: Path, idx: int) -> Path:
    """Get frame directory (may be flat structure)."""
    p = masks_root / str(idx)
    return p if p.is_dir() else masks_root


# -------------------------
# Name mapping / parsing
# -------------------------

def load_name_map_json(p: Optional[Path]) -> Dict[str, str]:
    """
    name_map format examples:

    Option A (simple map):
    {
      "knob": "handle",
      "buttons": "button",
      "socket": "outlet"
    }

    Option B (more explicit):
    {
      "map": { "knob":"handle" },
      "groups": [
        ["handle","knob"],
        ["button","buttons","switch"]
      ]
    }
    """
    if p is None:
        return {}

    if not p.exists():
        raise FileNotFoundError(f"name_map_json not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}

    if isinstance(obj, dict) and "map" in obj and isinstance(obj["map"], dict):
        for k, v in obj["map"].items():
            if k is None or v is None:
                continue
            mapping[normalize_name(str(k))] = normalize_name(str(v))

    if isinstance(obj, dict) and "groups" in obj and isinstance(obj["groups"], list):
        for g in obj["groups"]:
            if not isinstance(g, list) or len(g) == 0:
                continue
            canon = normalize_name(str(g[0]))
            for item in g:
                mapping[normalize_name(str(item))] = canon

    if isinstance(obj, dict) and "map" not in obj and "groups" not in obj:
        for k, v in obj.items():
            mapping[normalize_name(str(k))] = normalize_name(str(v))

    def resolve(x: str) -> str:
        seen = set()
        cur = x
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        return cur

    mapping = {k: resolve(v) for k, v in mapping.items()}
    return mapping


def canon_name(name: str, name_map: Dict[str, str]) -> str:
    n = normalize_name(name)
    return name_map.get(n, n)


def parse_mask_type_name_inst(stem: str) -> Tuple[str, str, str]:
    """
    Parse mask filename to extract type, name, inst_id.

    Expected patterns like:
      CTX__door__000__area428081
      INT__handle__001__area3190
    or underscore variants:
      CTX_door_000_area123

    Returns:
      typ: "object" or "affordance"
      name: normalized semantic name (string)
      inst_id: zero-padded string if present, else "000"
    """
    s = stem.strip()

    if s.startswith("CTX"):
        typ = "object"
        prefix = "CTX"
    elif s.startswith("INT"):
        typ = "affordance"
        prefix = "INT"
    else:
        typ = "object"
        prefix = None

    if "__" in s:
        parts = [p for p in s.split("__") if p]
    else:
        parts = [p for p in s.split("_") if p]

    # remove prefix token
    if prefix and parts and parts[0] == prefix:
        parts = parts[1:]
    elif prefix and parts and parts[0].startswith(prefix):
        parts[0] = parts[0].replace(prefix, "", 1)

    # extract inst_id if appears as a pure number token
    inst_id = "000"
    name_tokens: List[str] = []
    for p in parts:
        if re.fullmatch(r"\d+", p):
            inst_id = p.zfill(3)
            break
        if re.fullmatch(r"area\d+", p, flags=re.IGNORECASE):
            break
        if not p:
            continue
        name_tokens.append(p)

    name = " ".join(name_tokens).strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "unknown"

    name = normalize_name(name)
    return typ, name, inst_id


# -------------------------
# Depth helpers
# -------------------------

def load_depth(depth_path: Path) -> np.ndarray:
    """
    Load depth map into float32 (H,W) in RAW units.
    Supports: .npy, .png/.jpg/.tif...
    """
    suf = depth_path.suffix.lower()
    if suf == ".npy":
        d = np.load(str(depth_path))
        if d.ndim == 3:
            d = d[..., 0]
        return np.asarray(d, dtype=np.float32)

    if cv2 is not None:
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read depth: {depth_path}")
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)

    from PIL import Image
    d = np.array(Image.open(depth_path))
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def find_depth_dir_auto(seq_dir: Path) -> Optional[Path]:
    """Best-effort auto-find common depth folders under seq_dir."""
    cands = [
        seq_dir / "hires_wide_depth",
        seq_dir / "hires_depth",
        seq_dir / "depth",
        seq_dir / "hires_wide" / "depth",
        seq_dir / "hires_wide" / "depths",
        seq_dir / "depths",
    ]
    for p in cands:
        if p.exists() and p.is_dir():
            return p
    return None


def list_depth_files(depth_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy")
    files = [p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files, key=lambda p: natural_key(p.name))
    return files


# -------------------------
# Scene/sequence discovery
# -------------------------

def find_first_sequence_dir(scene_root: Path) -> Path:
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")

    seq_dirs = [p for p in scene_root.iterdir() if p.is_dir()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence dirs under: {scene_root}")

    numeric = [p for p in seq_dirs if re.fullmatch(r"\d+", p.name)]
    if numeric:
        return sorted(numeric, key=lambda p: natural_key(p.name))[0]

    return sorted(seq_dirs, key=lambda p: natural_key(p.name))[0]


def find_transform_npy(seq_dir: Path) -> Optional[Path]:
    cands = sorted(list(seq_dir.glob("*_transform.npy")), key=lambda p: natural_key(p.name))
    return cands[0] if cands else None


def find_traj_path(seq_dir: Path) -> Path:
    p = seq_dir / "hires_poses.traj"
    if not p.exists():
        raise FileNotFoundError(f"Trajectory not found: {p}")
    return p


def find_intr_dir(seq_dir: Path) -> Path:
    p = seq_dir / "hires_wide_intrinsics"
    if not p.exists():
        raise FileNotFoundError(f"Intrinsics dir not found: {p}")
    return p


# -------------------------
# Point cloud loading
# -------------------------

def load_pcd_points(pcd_path: Path) -> np.ndarray:
    """
    Load point cloud from: .npy, .npz, .ply, .pcd
    """
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

    suf = pcd_path.suffix.lower()

    if suf == ".npy":
        pts = np.load(str(pcd_path)).astype(np.float32)

    elif suf == ".npz":
        z = np.load(str(pcd_path))
        key = None
        for k in ["points", "xyz", "pcd"]:
            if k in z:
                key = k
                break
        if key is None:
            raise ValueError(f"NPZ has no point key. Available: {list(z.keys())[:10]}")
        pts = np.asarray(z[key], dtype=np.float32)

    elif suf in [".ply", ".pcd"]:
        try:
            import open3d as o3d
        except ImportError as e:
            raise RuntimeError("Need open3d for .ply/.pcd files. Install: pip install open3d") from e
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported format: {pcd_path}")

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Point cloud must be (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {pcd_path}")

    return pts


def auto_find_pcd(seq_dir: Path) -> Optional[Path]:
    cands = []
    for pat in ["*.ply", "*.pcd", "*.npy", "*.npz"]:
        cands += list(seq_dir.glob(pat))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None

    def score(p: Path):
        n = p.name.lower()
        s = 0
        if "point" in n:
            s += 2
        if "pcd" in n:
            s += 2
        if "scene" in n:
            s += 1
        return (-s, natural_key(p.name))

    cands = sorted(cands, key=score)
    return cands[0]
