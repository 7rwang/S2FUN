import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# headless backend (only used if save_vis=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

import torch.distributed as dist

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------- visualization palette ----------
_CMAP = plt.get_cmap("tab20")
_COLORS = [_CMAP(i) for i in range(20)]


# ---------------- distributed env ----------------

def get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def init_distributed_if_needed():
    """
    Make seg.py runnable under:
      - python (single GPU/CPU)
      - torchrun (multi-GPU)

    torchrun will set RANK/WORLD_SIZE/LOCAL_RANK.
    """
    rank, world_size, local_rank = get_dist_info()

    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1.")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def dist_barrier_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------- helpers ----------------

def natural_key(p: Path):
    parts = re.split(r"(\d+)", p.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _num_items(x) -> int:
    """
    Robust length for list/tuple/np/torch containers.
    Critical: torch.Tensor cannot be used as bool; empty tensor triggers RuntimeError on `if not x`.
    """
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        # (N,...) -> N; scalar -> numel
        return int(x.shape[0]) if x.ndim >= 1 else int(x.numel())
    try:
        return len(x)
    except Exception:
        return 0


def _normalize_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def split_prompts(prompt_str: str, delim: str = ";") -> List[str]:
    """
    Split 'wall switch; pull chain' -> ['wall switch','pull chain']
    Also trims and drops empties; stable de-dup (case-insensitive).
    """
    if prompt_str is None:
        return []
    s = str(prompt_str).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(delim)]
    parts = [_normalize_ws(p) for p in parts if _normalize_ws(p)]
    out: List[str] = []
    seen = set()
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def slugify_prompt(s: str, max_len: int = 80) -> str:
    s = _normalize_ws(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        s = "prompt"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def mask_area(mask, thr=0.5) -> int:
    m = _to_numpy(mask)
    if m is None:
        return 0
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    return int((m > thr).sum())


def save_mask_png(mask, path: Path, thr=0.5):
    m = _to_numpy(mask)
    if m is None:
        return
    if m.ndim == 3:
        m = np.squeeze(m, 0)
    m = (m > thr).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m, mode="L").save(path)


def save_vis(img_pil: Image.Image, overlays: List[dict], out_path: Path, thr=0.5, title: Optional[str] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_pil)
    ax.axis("off")
    if title:
        ax.set_title(title)

    for i, ov in enumerate(overlays):
        color = _COLORS[i % len(_COLORS)]
        mask = _to_numpy(ov.get("mask"))
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = np.squeeze(mask, 0)

        rgba = np.array([color[0], color[1], color[2], 0.45], dtype=np.float32)
        ax.imshow((mask > thr)[..., None] * rgba)

        box = ov.get("box", None)
        if box is not None:
            b = _to_numpy(box)
            if b is not None and len(b) == 4:
                if hasattr(b, "tolist"):
                    b = b.tolist()
                x0, y0, x1, y1 = map(float, b)
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=1.5,
                        edgecolor=(color[0], color[1], color[2], 1.0),
                        facecolor="none",
                    )
                )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- robust SAM3 state handling ----------------

def set_image_get_state(processor: Sam3Processor, img: Image.Image):
    st = processor.set_image(img)
    if st is not None:
        return st

    for attr in ("state", "_state", "inference_state", "_inference_state", "image_state", "_image_state"):
        if hasattr(processor, attr):
            v = getattr(processor, attr)
            if v is not None:
                return v

    raise ValueError("set_image returned None and no internal state attribute was found.")


# ---------------- data discovery ----------------

def find_first_sequence_image_dir(data_root: Path, scene_id: int, image_subdir: str) -> Optional[Path]:
    """
    data_root/<scene_id>/<sequence_id>/<image_subdir>
    Returns FIRST sequence (natural sort).
    """
    scene_dir = data_root / str(scene_id)
    if not scene_dir.exists():
        return None

    sequences = sorted([p for p in scene_dir.iterdir() if p.is_dir()], key=natural_key)
    if not sequences:
        return None

    first_seq = sequences[0]
    img_dir = first_seq / image_subdir
    if not img_dir.exists() or not img_dir.is_dir():
        return None
    return img_dir


def list_images(d: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(list(d.glob(ext)))
    return sorted(imgs, key=natural_key)


# ---------------- score extraction (STRICT) ----------------

def get_mask_scores_strict(out: dict, masks) -> List[float]:
    """
    STRICT requirement from you:
      - score MUST come from model output, specifically out["scores"]
      - NO fallback from mask statistics
    """
    if out is None:
        raise RuntimeError("SAM3 output is None; cannot read scores.")
    if "scores" not in out or out["scores"] is None:
        raise KeyError(
            "SAM3 output does not contain 'scores' (or it's None). "
            "You asked to forbid fallback scores, so we must stop here."
        )

    n = _num_items(masks)
    if n <= 0:
        return []

    s = _to_numpy(out["scores"])
    if s is None:
        raise RuntimeError("out['scores'] exists but could not be converted to numpy.")
    s = np.asarray(s).reshape(-1).tolist()

    if len(s) < n:
        raise RuntimeError(f"out['scores'] length ({len(s)}) < #masks ({n}). Refusing to guess/align.")

    return [float(x) for x in s[:n]]


# ---------------- core per-scene inference ----------------

def _run_sharded_best_per_annot(
    processor: Sam3Processor,
    scene_id: int,
    image_dir: Path,
    annot2prompt: Dict[str, str],
    out_root: Path,
    thr: float,
    save_vis_flag: bool,
    rank: int,
    world_size: int,
    prompt_delim: str = ";",
) -> Path:
    """
    For this rank: iterate owned frames, maintain local best per annot_id
    across ALL split prompts and ALL frames.
    Saves local json describing best picks and corresponding pngs under tmp folder.
    Returns path to local json.
    """
    scene_out = out_root / str(scene_id)
    tmp_root = scene_out / "tmp_rank_best"
    tmp_root.mkdir(parents=True, exist_ok=True)

    annot2prompts: Dict[str, List[str]] = {aid: split_prompts(p, delim=prompt_delim) for aid, p in annot2prompt.items()}
    local_best: Dict[str, Dict[str, Any]] = {}

    images = list_images(image_dir)
    num_frames = len(images)
    local_indices = [i for i in range(num_frames) if (i % world_size) == rank]

    pbar = tqdm(
        local_indices,
        total=len(local_indices),
        desc=f"[R{rank}] frames(best-per-annot; split-prompts)",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for fi in pbar:
        img_path = images[fi]
        img = Image.open(img_path).convert("RGB")
        state = set_image_get_state(processor, img)

        overlays = []

        for aid, prompts in annot2prompts.items():
            if not prompts:
                continue

            frame_best_m = None
            frame_best_box = None
            frame_best_s = None
            frame_best_prompt = None

            for prompt in prompts:
                out = processor.set_text_prompt(state=state, prompt=prompt)
                masks = out.get("masks", None) if out else None
                boxes = out.get("boxes", None) if out else None

                if _num_items(masks) == 0:
                    continue

                # STRICT scores: only from out["scores"]
                scores = get_mask_scores_strict(out, masks)

                best_m = None
                best_box = None
                best_s = None

                # iterate masks (works for list[Tensor] or Tensor batch)
                if isinstance(masks, torch.Tensor):
                    mask_iter = (masks[i] for i in range(_num_items(masks)))
                else:
                    mask_iter = iter(masks)

                for mi, m in enumerate(mask_iter):
                    if mask_area(m, thr) <= 0:
                        continue
                    s = float(scores[mi])
                    if (best_s is None) or (s > best_s):
                        best_s = s
                        best_m = m
                        if boxes is not None and mi < _num_items(boxes):
                            best_box = boxes[mi]
                        else:
                            best_box = None

                if best_m is None:
                    continue

                if (frame_best_s is None) or (float(best_s) > float(frame_best_s)):
                    frame_best_s = float(best_s)
                    frame_best_m = best_m
                    frame_best_box = best_box
                    frame_best_prompt = prompt

            if frame_best_m is None:
                continue

            prev = local_best.get(aid)
            if (prev is None) or (float(frame_best_s) > float(prev["score"])):
                out_dir = tmp_root / f"rank{rank:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)

                pslug = slugify_prompt(frame_best_prompt or "prompt")
                png_path = out_dir / f"{aid}__{pslug}__frame{fi:06d}__score{frame_best_s:.6f}.png"
                save_mask_png(frame_best_m, png_path, thr=thr)

                local_best[aid] = {
                    "annot_id": aid,
                    "prompt_used": frame_best_prompt,
                    "prompt_slug": pslug,
                    "score": float(frame_best_s),
                    "frame": int(fi),
                    "png_path": str(png_path),
                }

            if save_vis_flag:
                overlays.append({"mask": frame_best_m, "box": frame_best_box})

        if save_vis_flag:
            vis_out = (out_root / str(scene_id)) / "vis_best" / f"{fi:06d}.png"
            save_vis(
                img_pil=img,
                overlays=overlays,
                out_path=vis_out,
                thr=thr,
                title=f"{scene_id} frame={fi:06d} rank={rank} (best-per-annot; split-prompts)",
            )

    local_json = tmp_root / f"local_best_rank{rank:02d}.json"
    local_json.write_text(json.dumps(local_best, indent=2, ensure_ascii=False), encoding="utf-8")
    return local_json


def _merge_rank_bests(scene_out: Path, world_size: int) -> Dict[str, Dict[str, Any]]:
    """
    Rank0: merge tmp_rank_best/local_best_rankXX.json, pick best score per annot_id,
    copy png to final output folder.

    Files:
      anno_best_masks/<annot_id>.png                      (for downstream)
      anno_best_masks/<annot_id>__<prompt_slug>.png       (human-readable)
    """
    tmp_root = scene_out / "tmp_rank_best"
    final_dir = scene_out / "anno_best_masks"
    final_dir.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Dict[str, Any]] = {}

    for r in range(world_size):
        p = tmp_root / f"local_best_rank{r:02d}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for aid, meta in data.items():
            s = float(meta.get("score", 0.0))
            if (aid not in merged) or (s > float(merged[aid]["score"])):
                merged[aid] = meta

    out_map: Dict[str, Dict[str, Any]] = {}
    for aid, meta in merged.items():
        src = Path(meta["png_path"])
        if not src.exists():
            continue

        pslug = str(meta.get("prompt_slug") or "prompt")
        dst_named = final_dir / f"{aid}__{pslug}.png"
        dst_plain = final_dir / f"{aid}.png"

        shutil.copyfile(src, dst_named)
        shutil.copyfile(src, dst_plain)

        out_map[aid] = {
            "annot_id": aid,
            "prompt_used": meta.get("prompt_used", ""),
            "prompt_slug": pslug,
            "score": float(meta.get("score", 0.0)),
            "frame": int(meta.get("frame", -1)),
            "mask_png": str(dst_plain),
            "mask_png_named": str(dst_named),
        }

    (scene_out / "annot2bestmask.json").write_text(
        json.dumps(out_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_map


# ---------------- public API for run.py ----------------

def seg_scene(
    *,
    data_root: Union[str, Path],
    output_root: Union[str, Path],
    scene_id: int,
    annot2prompt: Dict[str, str],
    image_subdir: str = "hires_wide",
    thr: float = 0.5,
    save_vis: bool = False,
    device: Optional[str] = None,
    prompt_delim: str = ";",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Multi-GPU ready (torchrun):
      - init process group if WORLD_SIZE>1
      - shard frames by (frame_idx % world_size == rank)
      - rank0 merges best-per-annot and writes annot2bestmask.json

    STRICT:
      - score ONLY from SAM3 output["scores"]
      - if scores missing/misaligned -> raise
    """
    rank, world_size, local_rank = init_distributed_if_needed()

    if torch.cuda.is_available():
        if device is None:
            device = f"cuda:{local_rank}"
        # honor explicit device but still align cuda device for this process
        torch.cuda.set_device(int(device.split(":")[-1]) if ":" in device else local_rank)

    data_root = Path(data_root)
    out_root = Path(output_root)

    img_dir = find_first_sequence_image_dir(data_root, scene_id, image_subdir)
    if img_dir is None:
        raise RuntimeError(
            f"scene {scene_id}: cannot find first sequence with subdir '{image_subdir}' under {data_root}/{scene_id}"
        )

    scene_out = out_root / str(scene_id)
    scene_out.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        (scene_out / "annot2prompt.json").write_text(
            json.dumps(annot2prompt, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        annot2prompts_dump = {aid: split_prompts(p, delim=prompt_delim) for aid, p in annot2prompt.items()}
        (scene_out / "annot2prompts_split.json").write_text(
            json.dumps(annot2prompts_dump, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        meta = {
            "scene_id": int(scene_id),
            "sequence_used": img_dir.parent.name,
            "image_subdir": img_dir.name,
            "num_images": len(list_images(img_dir)),
            "num_annots": len(annot2prompt),
            "world_size": int(world_size),
            "sharding": "frame_idx % world_size == rank",
            "thr": float(thr),
            "prompt_delim": prompt_delim,
            "score_source": "SAM3 output['scores'] (STRICT; no fallback)",
        }
        (scene_out / "seg_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # build model per process
    model = build_sam3_image_model()
    try:
        model = model.to(torch.device(device)) if device is not None else model
    except Exception:
        pass
    model.eval()
    processor = Sam3Processor(model)

    with torch.no_grad():
        _run_sharded_best_per_annot(
            processor=processor,
            scene_id=scene_id,
            image_dir=img_dir,
            annot2prompt=annot2prompt,
            out_root=out_root,
            thr=thr,
            save_vis_flag=save_vis,
            rank=rank,
            world_size=world_size,
            prompt_delim=prompt_delim,
        )

    dist_barrier_if_needed()

    if rank == 0:
        out_map = _merge_rank_bests(scene_out=scene_out, world_size=world_size)
        return out_map

    return None
