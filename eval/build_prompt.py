import ast
import csv
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# -------------------------
# Parsing helpers
# -------------------------

def _parse_list_cell(cell: Any) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []

    # Try JSON first
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            return [val.strip()] if val.strip() else []
    except Exception:
        pass

    # Fallback: Python literal eval
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            return [val.strip()] if val.strip() else []
    except Exception:
        pass

    return [s] if s else []


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _clean_interactive_phrase(phrase: str) -> str:
    """
    Keep ONLY interactive_objects, but apply a special rule:
      - If phrase mentions socket + opening -> reduce to "socket"
    Examples:
      "socket opening" -> "socket"
      "socket-opening" -> "socket"
      "opening socket" -> "socket"
      "the socket opening" -> "socket"  (still becomes socket)
    """
    p = _normalize_ws(phrase).lower()
    if not p:
        return ""

    # treat underscores/hyphens as spaces for matching
    tokens = re.split(r"[ \t_\-]+", p)
    token_set = set(t for t in tokens if t)

    # Special case: socket + opening => socket
    if "socket" in token_set and "opening" in token_set:
        return "socket"

    # Otherwise keep original normalized (but still lower + normalized spaces)
    return p


# -------------------------
# Scene JSON discovery + parsing
# -------------------------

def _find_scene_jsons(scene_dir: Path) -> Tuple[Path, Path]:
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    jsons = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]

    desc_cands = [p for p in jsons if ("desc" in p.name.lower())]
    annot_cands = [p for p in jsons if ("annot" in p.name.lower())]

    if len(desc_cands) == 0:
        raise FileNotFoundError(f"No descriptions json found in: {scene_dir}")
    if len(annot_cands) == 0:
        raise FileNotFoundError(f"No annotations json found in: {scene_dir}")

    def _prefer(paths: List[Path], key: str) -> Path:
        exact = [p for p in paths if key in p.name.lower()]
        if len(exact) == 1:
            return exact[0]
        if len(paths) == 1:
            return paths[0]
        raise ValueError(f"Ambiguous {key} json candidates in {scene_dir}: {[p.name for p in paths]}")

    desc_json = _prefer(desc_cands, "descriptions")
    annot_json = _prefer(annot_cands, "annotations")
    return desc_json, annot_json


def _load_desc_to_annots(descriptions_json_path: Path) -> Dict[str, List[str]]:
    with descriptions_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    desc2annots: Dict[str, List[str]] = {}
    for d in data.get("descriptions", []):
        desc_id = str(d.get("desc_id", "")).strip()
        ann = d.get("annot_id", [])
        if isinstance(ann, str):
            ann_list = [ann]
        elif isinstance(ann, list):
            ann_list = [str(x).strip() for x in ann if str(x).strip()]
        else:
            ann_list = []
        if desc_id:
            desc2annots[desc_id] = ann_list
    return desc2annots


def _load_scene_annot_ids(annotations_json_path: Path) -> set:
    """
    annotations.json structure:
      {"visit_id":"420673","annotations":[{"annot_id":"...","indices":[...]}, ...]}
    """
    with annotations_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = set()
    for a in data.get("annotations", []):
        aid = str(a.get("annot_id", "")).strip()
        if aid:
            ids.add(aid)
    return ids


# -------------------------
# Main API
# -------------------------

def build_annotid_to_prompt_from_root(
    data_root: Union[str, Path],
    csv_path: Union[str, Path],
    *,
    out_json_path: Optional[Union[str, Path]] = None,
    join_delim: str = "; ",
    scene_id_filter: Optional[Union[str, int]] = None,
    filter_to_scene_annotations: bool = True,
    debug_scene_stats: bool = False,
) -> Dict[str, str]:
    """
    Build mapping annot_id -> prompt.

    IMPORTANT CHANGE (per your request):
      - Prompt uses ONLY interactive_objects (NO contextual_object merging).
      - Special case: "socket opening" => "socket"
    """

    data_root = Path(data_root)
    csv_path = Path(csv_path)

    scene_id_filter_str = None
    if scene_id_filter is not None:
        scene_id_filter_str = str(scene_id_filter).strip()

    # Cache per-scene:
    scene_desc_cache: Dict[str, Dict[str, List[str]]] = {}
    scene_annotid_cache: Dict[str, set] = {}

    annot_prompts_order: Dict[str, List[str]] = defaultdict(list)
    annot_prompts_seen: Dict[str, set] = defaultdict(set)

    dbg_csv_desc_counter: Dict[str, Counter] = defaultdict(Counter)
    dbg_hit_counter: Dict[str, int] = defaultdict(int)
    dbg_miss_counter: Dict[str, int] = defaultdict(int)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "description_id", "interactive_objects"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            scene_id = str(row.get("scene_id", "")).strip()
            if not scene_id:
                continue

            if scene_id_filter_str is not None and scene_id != scene_id_filter_str:
                continue

            desc_id = str(row.get("description_id", "")).strip()
            if not desc_id:
                continue

            inter_list = _parse_list_cell(row.get("interactive_objects", ""))

            # Load per-scene mapping once
            if scene_id not in scene_desc_cache:
                scene_dir = data_root / scene_id
                desc_json, annot_json = _find_scene_jsons(scene_dir)
                scene_desc_cache[scene_id] = _load_desc_to_annots(desc_json)
                scene_annotid_cache[scene_id] = _load_scene_annot_ids(annot_json)

            desc2annots = scene_desc_cache[scene_id]
            annots = desc2annots.get(desc_id, [])

            if debug_scene_stats:
                dbg_csv_desc_counter[scene_id][desc_id] += 1
                if annots:
                    dbg_hit_counter[scene_id] += 1
                else:
                    dbg_miss_counter[scene_id] += 1

            if not annots:
                continue

            # ---- PROMPT = ONLY interactive_objects ----
            desc_phrases: List[str] = []
            seen = set()

            for inter in inter_list:
                phrase = _clean_interactive_phrase(inter)
                if not phrase:
                    continue
                if phrase not in seen:
                    seen.add(phrase)
                    desc_phrases.append(phrase)

            # If interactive_objects empty/invalid -> skip (as per "only keep interactive")
            if not desc_phrases:
                continue

            prompt = join_delim.join(desc_phrases).strip()
            if not prompt:
                continue

            # Optionally filter annot_id by annotations.json
            valid_annots = scene_annotid_cache.get(scene_id, set()) if filter_to_scene_annotations else None

            for annot_id in annots:
                annot_id = str(annot_id).strip()
                if not annot_id:
                    continue
                if valid_annots is not None and annot_id not in valid_annots:
                    continue

                if prompt not in annot_prompts_seen[annot_id]:
                    annot_prompts_seen[annot_id].add(prompt)
                    annot_prompts_order[annot_id].append(prompt)

    annot2prompt: Dict[str, str] = {}
    for annot_id, prompts in annot_prompts_order.items():
        annot2prompt[annot_id] = join_delim.join(prompts)

    if debug_scene_stats:
        for sid in sorted(dbg_csv_desc_counter.keys()):
            total_rows = sum(dbg_csv_desc_counter[sid].values())
            uniq_desc = len(dbg_csv_desc_counter[sid])
            hit = dbg_hit_counter[sid]
            miss = dbg_miss_counter[sid]
            print("=" * 80)
            print(f"[DEBUG] scene_id={sid}")
            print(f"  CSV rows (non-empty desc_id): {total_rows}")
            print(f"  Unique CSV description_id    : {uniq_desc}")
            print(f"  Rows with desc_id HIT in JSON: {hit}")
            print(f"  Rows with desc_id MISS in JSON: {miss}")
            if miss > 0:
                miss_set = set(dbg_csv_desc_counter[sid].keys()) - set(scene_desc_cache.get(sid, {}).keys())
                if miss_set:
                    top_miss = sorted(
                        [(d, dbg_csv_desc_counter[sid][d]) for d in miss_set],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    print("  Example missing desc_id (top 10 by freq):")
                    for d, cnt in top_miss:
                        print(f"    {d}  x{cnt}")
            print("=" * 80)

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(annot2prompt, f, ensure_ascii=False, indent=2)

    return annot2prompt
