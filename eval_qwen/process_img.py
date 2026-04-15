#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples:

# Single frame (1-based)
python process_img.py \
  --scene-graph /nas/qirui/sam3/S2FUN/eval_qwen/421254/scene_graph.json \
  --data-root /nas/qirui/scenefun3d/data \
  --frame-idx 50 --one-based \
  --out /nas/qirui/tmp/sg_proj.jpg \
  --label id_name \
  --pose-type auto

# Multiple frames (mix of single + range; 1-based)
python process_img.py \
  --scene-graph /nas/qirui/sam3/S2FUN/eval_qwen/421254/scene_graph.json \
  --data-root /nas/qirui/scenefun3d/data \
  --frame-idx 36 495 500-520 --one-based \
  --out /nas/qirui/tmp/sg_proj.jpg \
  --label id_name \
  --pose-type auto

Output naming:
- If --out is /path/sg_proj.jpg, outputs become:
  /path/sg_proj_f0036.jpg, /path/sg_proj_f0495.jpg, /path/sg_proj_f0500.jpg ...
"""

import argparse, json, re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def numeric_key(s: str):
    ns = re.findall(r"\d+", s)
    return (int(ns[-1]) if ns else 10**18, s)


def list_sorted_files(d: Path, exts: Optional[set] = None) -> List[Path]:
    files = [p for p in d.iterdir() if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]
    files.sort(key=lambda p: numeric_key(p.stem))
    return files


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def parse_intrinsics_line(line: str) -> Tuple[int, int, float, float, float, float]:
    # expected: W H fx fy cx cy (in one line, possibly extra tokens)
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line.strip())]
    if len(vals) < 6:
        raise ValueError(f"Bad intrinsics line (need 6 nums): {line}")
    W, H = int(vals[0]), int(vals[1])
    fx, fy, cx, cy = map(float, vals[2:6])
    return W, H, fx, fy, cx, cy


def load_intrinsics_from_dir(intr_dir: Path, frame: int) -> np.ndarray:
    """
    Supports:
      - single .pincam or .txt containing one line: W H fx fy cx cy
      - per-frame .pincam/.txt files; select by numeric-sorted index
    """
    intr_files = [
        p for p in intr_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".pincam"}
    ]
    if not intr_files:
        raise FileNotFoundError(f"No .txt or .pincam intrinsics in {intr_dir}")

    intr_files.sort(key=lambda p: numeric_key(p.stem))

    if len(intr_files) == 1:
        line = intr_files[0].read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0]
    else:
        if frame >= len(intr_files):
            raise IndexError(f"frame={frame} but intr_files={len(intr_files)} in {intr_dir}")
        line = intr_files[frame].read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0]

    W, H, fx, fy, cx, cy = parse_intrinsics_line(line)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def parse_pose_line(line: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Confirmed:
      timestamp angle_axis_x angle_axis_y angle_axis_z translation_x translation_y translation_z
    angle-axis is rotation vector in radians; translation in meters.
    Returns:
      ts, rvec(3,), t(3,)
    """
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line.strip())]
    if len(vals) < 7:
        raise ValueError(f"Bad pose line (need 7 nums): {line}")
    ts = vals[0]
    rvec = np.array(vals[1:4], dtype=np.float64)   # radians
    t = np.array(vals[4:7], dtype=np.float64)      # meters
    return ts, rvec, t


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def project(K: np.ndarray, Tcw: np.ndarray, Pw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pw: (N,3) world
    Tcw: 4x4 world->cam
    """
    N = Pw.shape[0]
    Pw_h = np.concatenate([Pw, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
    Pc_h = (Tcw @ Pw_h.T).T
    Pc = Pc_h[:, :3]
    z = Pc[:, 2]
    x = Pc[:, 0] / (z + 1e-12)
    y = Pc[:, 1] / (z + 1e-12)
    uv = (K @ np.stack([x, y, np.ones_like(x)], axis=0)).T[:, :2]
    return uv, z


def score_uv(uv: np.ndarray, z: np.ndarray, W: int, H: int) -> Tuple[float, float, float]:
    okz = z > 1e-6
    inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    in_front_ratio = float(okz.mean()) if uv.size else 0.0
    in_img_ratio = float((okz & inimg).mean()) if uv.size else 0.0
    score = in_front_ratio * 0.7 + in_img_ratio * 0.3
    return score, in_front_ratio, in_img_ratio


def draw(img: np.ndarray, uv: np.ndarray, z: np.ndarray, labels: List[str]) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for (u, v), zz, lab in zip(uv, z, labels):
        if zz <= 1e-6:
            continue
        if not (0 <= u < W and 0 <= v < H):
            continue
        pt = (int(round(u)), int(round(v)))
        cv2.circle(out, pt, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(out, lab, (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def expand_frame_specs(specs: List[str]) -> List[int]:
    """
    Expand a list like: ["36", "495", "500-520"] into sorted unique ints.
    Accepts:
      - "N"
      - "A-B" (inclusive)
    Rejects comma-separated like "36,495" on purpose.
    """
    frames: List[int] = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if "," in s:
            raise ValueError(f"Do not use commas in --frame-idx. Use spaces: e.g. --frame-idx 36 495. Got: {s}")
        if "-" in s:
            a, b = s.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            frames.extend(list(range(a, b + 1)))
        else:
            frames.append(int(s))
    frames = sorted(set(frames))
    return frames


def make_out_path(base_out: Path, frame: int) -> Path:
    """
    base_out = /path/name.jpg
    -> /path/name_f0036.jpg
    """
    stem = base_out.stem
    suf = base_out.suffix if base_out.suffix else ".jpg"
    return base_out.parent / f"{stem}_f{frame:04d}{suf}"


def process_one_frame(
    sg: dict,
    data_root: Path,
    frame: int,
    one_based: bool,
    base_out: Path,
    label_mode: str,
    pose_type: str,
):
    scene_id = str(sg["scene_id"])
    seq_id = str(sg["sequence_id"])

    f0 = frame - 1 if one_based else frame
    if f0 < 0:
        raise ValueError(f"frame {frame} becomes negative after --one-based")

    seq_dir = data_root / scene_id / seq_id
    img_dir = seq_dir / "hires_wide"
    intr_dir = seq_dir / "hires_wide_intrinsics"
    traj_path = seq_dir / "hires_poses.traj"

    if not img_dir.exists():
        raise FileNotFoundError(f"missing image dir: {img_dir}")
    if not intr_dir.exists():
        raise FileNotFoundError(f"missing intr dir: {intr_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(f"missing traj: {traj_path}")

    images = list_sorted_files(img_dir, IMG_EXTS)
    if f0 >= len(images):
        raise IndexError(f"frame0={f0} but images={len(images)} in {img_dir}")

    img_path = images[f0]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {img_path}")
    H, W = img.shape[:2]

    K = load_intrinsics_from_dir(intr_dir, f0)

    lines = [ln.strip() for ln in traj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if f0 >= len(lines):
        raise IndexError(f"frame0={f0} but pose lines={len(lines)} in {traj_path}")

    ts, rvec, t = parse_pose_line(lines[f0])
    R, _ = cv2.Rodrigues(rvec)  # radians confirmed
    baseT = make_T(R, t)        # either Twc or Tcw depending on convention

    nodes = sg.get("nodes", [])
    Pw = np.array([n["position"] for n in nodes], dtype=np.float64)

    labels = []
    for n in nodes:
        nid = str(n.get("id", ""))
        if label_mode == "id":
            labels.append(nid)
        else:
            nm = n.get("name", "")
            labels.append(f"{nid}:{nm}" if nm else nid)

    # Only ambiguity: baseT is Twc or Tcw
    candidates = []
    if pose_type in ("auto", "Twc"):
        candidates.append(("Twc_inv->Tcw", np.linalg.inv(baseT)))
    if pose_type in ("auto", "Tcw"):
        candidates.append(("Tcw_as_is", baseT))

    best_tag = None
    best_s = -1e9
    best_uv = None
    best_z = None

    for tag, Tcw in candidates:
        uv, z = project(K, Tcw, Pw)
        s, in_front, in_img = score_uv(uv, z, W, H)
        print(f"[frame={frame}] [DEBUG] {tag}: score={s:.4f} in_front={in_front:.4f} in_img={in_img:.4f}")
        if s > best_s:
            best_s = s
            best_tag = tag
            best_uv, best_z = uv, z

    vis = draw(img, best_uv, best_z, labels)

    out_path = make_out_path(base_out, frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), vis):
        raise RuntimeError(f"failed to write: {out_path}")

    print(f"[frame={frame}] image={img_path}")
    print(f"[frame={frame}] pose_ts={ts}")
    print(f"[frame={frame}] pose_used={best_tag} best_score={best_s:.4f}")
    print(f"[frame={frame}] saved={out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-graph", required=True)
    ap.add_argument("--data-root", required=True)  # /nas/qirui/scenefun3d/data

    # accept multiple specs: N, A-B
    ap.add_argument(
        "--frame-idx",
        required=True,
        nargs="+",
        type=str,
        help="One or more frame indices. Use spaces, not commas. Supports ranges: 500-520.",
    )

    ap.add_argument("--one-based", action="store_true")
    ap.add_argument("--out", required=True, help="Base output path; will append _fXXXX.")
    ap.add_argument("--label", choices=["id", "id_name"], default="id_name")
    ap.add_argument("--pose-type", choices=["auto", "Twc", "Tcw"], default="auto",
                    help="If known, set Twc (cam->world) or Tcw (world->cam). Otherwise auto pick.")
    args = ap.parse_args()

    sg = load_json(Path(args.scene_graph))
    data_root = Path(args.data_root)
    base_out = Path(args.out)

    frames = expand_frame_specs(args.frame_idx)
    if not frames:
        raise ValueError("No frames parsed from --frame-idx")

    for f in frames:
        process_one_frame(
            sg=sg,
            data_root=data_root,
            frame=f,
            one_based=args.one_based,
            base_out=base_out,
            label_mode=args.label,
            pose_type=args.pose_type,
        )


if __name__ == "__main__":
    main()