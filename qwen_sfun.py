#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial-aware interface grounding for SceneFun3D.

Two modes:
- If (anchor_object != N/A) AND (spatial_relationship != N/A):
    -> find anchor node in scene graph, then use relation to pick ONE best interface candidate (top1).
- Else (both N/A):
    -> output TOP-3 interface candidates (multi-select).

Candidates can be either:
- type == "affordance"
- type == "object"

Fallback:
- If model returns nothing/invalid:
    -> use CSV 'interactive_objects' to retrieve candidates (top1 or top3 depending on mode)
    -> special rule: "socket opening" => keep only "socket" related nodes.

Outputs:
1) matches_scene{scene_id}.json
   - per row -> instruction + candidates (id,name,type,score,source)
2) candidate_ids_scene{scene_id}.json
   - per row -> instruction + candidate_ids (same length/order as candidates in matches)
   - ONLY these two fields per row

CLI unchanged.

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
python qwen_sfun.py \
  --scene_id 420673 \
  --scene_graph /nas/qirui/sam3/S2FUN/eval_qwen/0_5-4/dummy.json \
  --csv_path /nas/qirui/sam3/scenefun3d_ex/parse_result.csv \
  --out_dir /nas/qirui/sam3/S2FUN/qwen_grounding \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --scene_graph_mode more \
  --per_gpu_mem_gb 40 \
  --dtype bf16 

"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# IO
# ----------------------------

def load_scene_graph(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("nodes", None), list):
        return data["nodes"]
    raise ValueError(f"scene_graph json must be a list or dict with 'nodes': {p}")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for r in reader:
            rows.append(r)
    return rows


# ----------------------------
# Utilities
# ----------------------------

def extract_json_object(text: str) -> Optional[str]:
    """Extract first complete JSON object via brace matching."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def clamp01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def compact_scene_graph(scene_graph: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "full":
        return scene_graph
    out: List[Dict[str, Any]] = []
    for n in scene_graph:
        base: Dict[str, Any] = {
            "id": n.get("id", None),
            "type": n.get("type", None),
            "name": n.get("name", None),
            "parent_id": n.get("parent_id", None),
        }
        if mode == "more":
            if "position" in n:
                base["position"] = n.get("position")
            if "names" in n:
                base["names"] = n.get("names")
            if "inst_id" in n:
                base["inst_id"] = n.get("inst_id")
        out.append(base)
    return out


def get_first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _id_to_node_map(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m: Dict[int, Dict[str, Any]] = {}
    for n in nodes:
        i = n.get("id", None)
        if isinstance(i, int):
            m[i] = n
    return m


def _sanitize_id(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def is_na(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return (s == "") or (s.upper() == "N/A") or (s.lower() == "na") or (s.lower() == "none")


def parse_interactive_objects_field(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [_norm(str(t)) for t in x if str(t).strip()]
    s = str(x).strip()
    if not s or s.upper() == "N/A":
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            import ast
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [_norm(str(t)) for t in arr if str(t).strip()]
        except Exception:
            pass
    parts = None
    for sep in [",", ";", "|"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            break
    if parts is None:
        parts = [s]
    return [_norm(p) for p in parts if p.strip()]


def fallback_candidates_from_interactive(
    interactive_texts: List[str],
    scene_graph_nodes: List[Dict[str, Any]],
    id2node: Dict[int, Dict[str, Any]],
    valid_candidate_ids: set,
    topk: int = 3,
) -> List[Dict[str, Any]]:
    if not interactive_texts:
        return []

    wants_socket = any(("socket opening" in t) or (t == "socket opening") for t in interactive_texts)

    tokens = set(interactive_texts)
    if wants_socket:
        tokens.add("socket")
        tokens = {t for t in tokens if "opening" not in t}

    scored: List[Tuple[float, int]] = []
    for n in scene_graph_nodes:
        nid = n.get("id", None)
        if not isinstance(nid, int) or nid not in valid_candidate_ids:
            continue

        name = n.get("name", "") or ""
        name_n = _norm(name)

        if wants_socket and ("socket" not in name_n):
            continue

        score = 0.0
        for t in tokens:
            if not t:
                continue
            if t == name_n:
                score = max(score, 1.0)
            elif t in name_n:
                score = max(score, 0.85)
            else:
                ov = len(set(t.split()) & set(name_n.split()))
                if ov > 0:
                    score = max(score, 0.45 + 0.1 * ov)

        if score > 0.0:
            scored.append((score, nid))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen = set()
    for sc, nid in scored:
        if nid in seen:
            continue
        seen.add(nid)
        node = id2node[nid]
        out.append({
            "id": nid,
            "name": node.get("name"),
            "type": node.get("type"),
            "score": float(sc),
            "source": "fallback_interactive_objects",
        })
        if len(out) >= topk:
            break
    return out


# ----------------------------
# Prompts
# ----------------------------

def build_prompt_top3(scene_graph_nodes: List[Dict[str, Any]], instruction: str) -> str:
    schema = {"candidates": [{"id": 4, "name": "door_knob", "type": "affordance", "score": 0.92}]}
    return f"""
STRICT JSON only. Output exactly one JSON object matching the schema. No extra text.

Task:
Select up to TOP-3 best scene-graph nodes that represent the interaction interface/tool needed to execute the instruction.
Candidates may be either:
- type=="affordance" (preferred)
- type=="object"

Hard constraints:
- Every candidate.id MUST exist in scene graph.
- candidate.name MUST exactly equal that node's name.
- candidate.type MUST exactly equal that node's type.
- candidates length <= 3, sorted by score descending.
- If no suitable candidate exists, output candidates=[].

Schema:
{json.dumps(schema, ensure_ascii=False)}

Instruction:
{instruction}

Scene graph nodes:
{json.dumps(scene_graph_nodes, ensure_ascii=False)}
""".strip()


def build_prompt_spatial_top1(
    scene_graph_nodes: List[Dict[str, Any]],
    instruction: str,
    contextual_object: str,
    anchor_object: str,
    spatial_relationship: str,
) -> str:
    schema = {
        "anchor": {"id": 24, "name": "window", "type": "object"},
        "choice": {"id": 4, "name": "door_knob", "type": "affordance"},
        "score": 0.92
    }
    return f"""
STRICT JSON only. Output exactly one JSON object matching the schema. No extra text.

You MUST use (contextual_object, anchor_object, spatial_relationship) to disambiguate.

Given:
- contextual_object: {contextual_object}
- anchor_object: {anchor_object}
- spatial_relationship: {spatial_relationship}
- instruction: {instruction}

Task:
1) Find the best-matching anchor node in the scene graph for anchor_object.
2) Use spatial_relationship + positions to disambiguate the intended contextual object instance.
3) Output ONE best interface/tool node ("choice") needed to execute the instruction:
   - Prefer affordance that belongs to that contextual object (parent_id match if available).
   - Otherwise pick the best interface-like node (affordance or object).

Hard constraints:
- anchor.id MUST exist and (anchor.name, anchor.type) must exactly match the scene-graph node.
- choice.id MUST exist and (choice.name, choice.type) must exactly match the scene-graph node.
- score in [0,1].

If anchor cannot be grounded at all, set anchor=null and still pick best choice if possible.
If no reasonable choice exists, set choice=null and score=0.

Schema:
{json.dumps(schema, ensure_ascii=False)}

Scene graph nodes:
{json.dumps(scene_graph_nodes, ensure_ascii=False)}
""".strip()


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", type=int, required=True)
    ap.add_argument("--scene_graph", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--scene_graph_mode", type=str, default="more", choices=["basic", "more", "full"])
    ap.add_argument("--per_gpu_mem_gb", type=int, default=40)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"matches_scene{args.scene_id}.json"
    out_ids = out_dir / f"candidate_ids_scene{args.scene_id}.json"

    # dtype
    torch_dtype: Optional[torch.dtype]
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    # sharding/offload strategy
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device_map: Union[str, Dict[str, int]] = "auto"
        max_memory = {i: f"{args.per_gpu_mem_gb}GiB" for i in range(n_gpus)}
    else:
        device_map = "cpu"
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    mp_kwargs: Dict[str, Any] = dict(device_map=device_map, trust_remote_code=True)
    if max_memory is not None:
        mp_kwargs["max_memory"] = max_memory
    if args.load_in_4bit:
        mp_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        mp_kwargs["load_in_8bit"] = True
    else:
        mp_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **mp_kwargs)
    model.eval()

    # load scene graph
    scene_graph = load_scene_graph(Path(args.scene_graph))
    scene_graph_nodes = compact_scene_graph(scene_graph, args.scene_graph_mode)
    id2node = _id_to_node_map(scene_graph_nodes)
    valid_candidate_ids = set(id2node.keys())

    # load + filter csv
    all_rows = read_csv_rows(Path(args.csv_path))
    if not all_rows:
        raise RuntimeError(f"Empty CSV: {args.csv_path}")

    header = set(all_rows[0].keys())
    if "scene_id" not in header:
        raise ValueError(f"CSV missing required column ['scene_id']. Header={list(all_rows[0].keys())}")
    if ("original_prompt" not in header) and ("description_prompt" not in header):
        raise ValueError(
            "CSV missing required prompt column: need one of ['original_prompt', 'description_prompt']. "
            f"Header={list(all_rows[0].keys())}"
        )

    keep_rows: List[Dict[str, str]] = []
    for i, r in enumerate(all_rows):
        if str(r.get("scene_id", "")).strip() == str(args.scene_id):
            r2 = dict(r)
            r2["_row_index"] = str(i)
            keep_rows.append(r2)

    if not keep_rows:
        raise RuntimeError(f"No CSV rows found for scene_id={args.scene_id} in {args.csv_path}")

    pbar = tqdm(keep_rows, desc=f"scene {args.scene_id}")
    first_device = get_first_device(model)

    results: List[Dict[str, Any]] = []
    ids_only: List[Dict[str, Any]] = []

    for r in pbar:
        row_index = int(r["_row_index"])
        instruction = (r.get("original_prompt") or r.get("description_prompt") or "").strip()

        contextual_object = (r.get("contextual_object", "") or "").strip()
        anchor_object = (r.get("anchor_object", "") or "").strip()
        spatial_relationship = (r.get("spatial_relationship", "") or "").strip()
        interactive_objects_raw = r.get("interactive_objects", None)

        if not instruction:
            results.append({
                "row_index": row_index,
                "instruction": instruction,
                "candidates": [],
                "error": "empty_instruction",
            })
            ids_only.append({"instruction": instruction, "candidate_ids": []})
            continue

        has_spatial = (not is_na(anchor_object)) and (not is_na(spatial_relationship))

        # ---------- LLM ----------
        if has_spatial:
            prompt = build_prompt_spatial_top1(
                scene_graph_nodes=scene_graph_nodes,
                instruction=instruction,
                contextual_object=contextual_object if not is_na(contextual_object) else "None",
                anchor_object=anchor_object,
                spatial_relationship=spatial_relationship,
            )
        else:
            prompt = build_prompt_top3(scene_graph_nodes, instruction)

        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(chat_text, return_tensors="pt")
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )

        out_text = tokenizer.decode(
            gen_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        json_str = extract_json_object(out_text)

        candidates: List[Dict[str, Any]] = []
        err: Optional[str] = None

        if json_str is None:
            err = "non_json_output"
        else:
            try:
                obj = json.loads(json_str)
                if not isinstance(obj, dict):
                    raise ValueError("non_dict_json")

                if has_spatial:
                    choice = obj.get("choice", None)
                    score = clamp01(obj.get("score", None))
                    score = 0.0 if score is None else score

                    if isinstance(choice, dict):
                        nid = _sanitize_id(choice.get("id", None))
                        nname = choice.get("name", None)
                        ntype = choice.get("type", None)
                        nname = nname.strip() if isinstance(nname, str) else None
                        ntype = ntype.strip() if isinstance(ntype, str) else None

                        if nid is not None and nid in valid_candidate_ids:
                            true = id2node[nid]
                            if (nname == true.get("name")) and (ntype == true.get("type")):
                                candidates = [{
                                    "id": nid,
                                    "name": nname,
                                    "type": ntype,
                                    "score": float(score),
                                    "source": "llm_spatial_top1",
                                }]
                            else:
                                err = "invalid_choice_name_or_type"
                        else:
                            err = "invalid_choice_id"
                    else:
                        err = "missing_choice"

                else:
                    raw_cands = obj.get("candidates", [])
                    if not isinstance(raw_cands, list):
                        raise ValueError("candidates_not_list")

                    tmp: List[Dict[str, Any]] = []
                    for it in raw_cands[:3]:
                        if not isinstance(it, dict):
                            continue
                        nid = _sanitize_id(it.get("id", None))
                        nname = it.get("name", None)
                        ntype = it.get("type", None)
                        nname = nname.strip() if isinstance(nname, str) else None
                        ntype = ntype.strip() if isinstance(ntype, str) else None
                        sc = clamp01(it.get("score", None))
                        sc = 0.0 if sc is None else sc

                        if nid is None or nid not in valid_candidate_ids:
                            continue
                        true = id2node[nid]
                        if (nname != true.get("name")) or (ntype != true.get("type")):
                            continue

                        tmp.append({
                            "id": nid,
                            "name": nname,
                            "type": ntype,
                            "score": float(sc),
                            "source": "llm_top3",
                        })

                    tmp.sort(key=lambda x: float(x["score"]), reverse=True)
                    seen = set()
                    for x in tmp:
                        if x["id"] in seen:
                            continue
                        seen.add(x["id"])
                        candidates.append(x)
                        if len(candidates) >= 3:
                            break

            except Exception as e:
                err = f"bad_json:{str(e)}"
                candidates = []

        # ---------- FALLBACK ----------
        if len(candidates) == 0:
            inter_texts = parse_interactive_objects_field(interactive_objects_raw)
            fb_topk = 1 if has_spatial else 3
            fb = fallback_candidates_from_interactive(
                interactive_texts=inter_texts,
                scene_graph_nodes=scene_graph_nodes,
                id2node=id2node,
                valid_candidate_ids=valid_candidate_ids,
                topk=fb_topk,
            )
            if fb:
                candidates = fb
                err = (err + "+fallback") if err else ("llm_empty_used_fallback_top1" if has_spatial else "llm_empty_used_fallback_top3")

        rec = {
            "row_index": row_index,
            "instruction": instruction,
            "candidates": candidates,
        }
        if err:
            rec["error"] = err
        results.append(rec)

        # ---- ids-only: ONLY instruction + candidate_ids (same count/order as candidates) ----
        ids_only.append({
            "instruction": instruction,
            "candidate_ids": [c["id"] for c in candidates],
        })

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"scene_id": args.scene_id, "model": args.model, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with out_ids.open("w", encoding="utf-8") as f:
        json.dump(
            {"scene_id": args.scene_id, "model": args.model, "results": ids_only},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Done.\n"
        f"- Output matches JSON: {out_json}\n"
        f"- Output ids JSON: {out_ids}\n"
        f"- Rows processed: {len(keep_rows)}\n"
    )


if __name__ == "__main__":
    main()
