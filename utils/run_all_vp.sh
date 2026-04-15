#!/usr/bin/env bash
set -u
set -o pipefail

# ---------------------------------------------
# Batch run visual prompt generation (ALL nodes)
#   - ONLY TEXT (no bbox visible)
#   - NMS suppress overlapping bboxes (avoid text overlap)
#   - SOFT-FAIL: python exits 0 even if no_support/uncovered exist
#   - Bash inspects selected_frames.json and prints OK vs SOFT-FAIL
# ---------------------------------------------

PY_SCRIPT="/nas/qirui/sam3/S2FUN/utils/project_scene_graph_visual_prompt_bbox.py"

DATA_ROOT="/nas/qirui/scenefun3d/data"
BASE_OUT="/nas/qirui/sam3/S2FUN/eval_qwen/split_30scenes"

# -------------------------
# Tuning knobs
# -------------------------
FRAME_STRIDE=1
SAMPLE_POINTS_PER_NODE=2000   # 0 = use all points (slower but more stable)
MIN_INLIERS=10
BBOX_MARGIN=2
TEXT_SCALE=1.2
TEXT_THICKNESS=1
TEXT_ALPHA=0.95
TEXT_POS="center"

MIN_BBOX_VIS_RATIO=0.80
FRAME_GOOD_BBOX_FRAC=0.80

SUPPRESS_IOU=0.70
SCORE_MODE="inliers_x_vis"

ENSURE_COVER_ALL=1            # 1=on, 0=off
MAX_EXTRA_FRAMES=-1           # -1=no cap

# -------------------------
# Scenes to process
# -------------------------
SCENES=(
  421254
  421393
  421657
  422356
  422842
  423070
  423452
  423957
  434888
  434897
  435324
  435357
  437157
  437294
  438102
  438775
  455342
  466092
  466112
  466162
  466437
  466841
  466876
  466880
  466916
  467115
  467139
  467330
  468282
  468476
)

# -------------------------
# Logs
# -------------------------
LOG_DIR="${BASE_OUT}/visual_prompt_logs_softfail"
mkdir -p "$LOG_DIR"

# -------------------------
# Helpers
# -------------------------
json_len_key() {
  # usage: json_len_key /path/to/report.json key_name
  local P="$1"
  local K="$2"
  python - "$P" "$K" <<'PY'
import json, sys
p = sys.argv[1]
k = sys.argv[2]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
v = d.get(k, [])
print(len(v) if isinstance(v, list) else 0)
PY
}

echo "[INFO] Python script: $PY_SCRIPT"
echo "[INFO] Data root:      $DATA_ROOT"
echo "[INFO] Base out:       $BASE_OUT"
echo "[INFO] Scenes:         ${#SCENES[@]}"
echo

for SID in "${SCENES[@]}"; do
  echo "============================================================"
  echo "[SCENE] ${SID}"
  echo "============================================================"

  SG_JSON="${BASE_OUT}/${SID}/scene_graph.json"
  PI_JSON="${BASE_OUT}/${SID}/scene_graph_point_indices.json"

  OUT_DIR="${BASE_OUT}/${SID}/visual_prompt_all_vis80_nms"
  mkdir -p "$OUT_DIR"

  if [[ ! -f "$SG_JSON" ]]; then
    echo "[WARN] missing scene_graph.json: $SG_JSON"
    echo "[SKIP] ${SID}"
    echo
    continue
  fi
  if [[ ! -f "$PI_JSON" ]]; then
    echo "[WARN] missing scene_graph_point_indices.json: $PI_JSON"
    echo "[SKIP] ${SID}"
    echo
    continue
  fi

  LOG_FILE="${LOG_DIR}/${SID}.log"

  set +e
  python "$PY_SCRIPT" \
    --scene_graph_json "$SG_JSON" \
    --point_indices_json "$PI_JSON" \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR" \
    --node_type all \
    --frame_stride "$FRAME_STRIDE" \
    --sample_points_per_node "$SAMPLE_POINTS_PER_NODE" \
    --min_inliers "$MIN_INLIERS" \
    --bbox_margin "$BBOX_MARGIN" \
    --text_scale "$TEXT_SCALE" \
    --text_thickness "$TEXT_THICKNESS" \
    --text_alpha "$TEXT_ALPHA" \
    --text_pos "$TEXT_POS" \
    --min_bbox_vis_ratio "$MIN_BBOX_VIS_RATIO" \
    --frame_good_bbox_frac "$FRAME_GOOD_BBOX_FRAC" \
    --suppress_iou "$SUPPRESS_IOU" \
    --score_mode "$SCORE_MODE" \
    $( [[ "$ENSURE_COVER_ALL" == "1" ]] && echo "--ensure_cover_all" ) \
    --max_extra_frames "$MAX_EXTRA_FRAMES" \
    --clear_out \
    > "$LOG_FILE" 2>&1
  RC=$?
  set -e

  # If python truly crashed, show FAIL
  if [[ $RC -ne 0 ]]; then
    echo "[FAIL] ${SID} (exit=$RC). See log: $LOG_FILE"
    echo
    continue
  fi

  REPORT_JSON="$OUT_DIR/selected_frames.json"
  if [[ ! -f "$REPORT_JSON" ]]; then
    echo "[WARN] ${SID} finished but missing report: $REPORT_JSON"
    echo "  log: $LOG_FILE"
    echo
    continue
  fi

  NS=$(json_len_key "$REPORT_JSON" "no_support_node_ids")
  UC=$(json_len_key "$REPORT_JSON" "uncovered_node_ids")

  if [[ "$NS" -gt 0 || "$UC" -gt 0 ]]; then
    echo "[SOFT-FAIL] ${SID}  no_support=${NS}  uncovered=${UC}"
  else
    echo "[OK] ${SID} (full covered)"
  fi

  echo "  out:    $OUT_DIR/frames"
  echo "  report: $REPORT_JSON"
  echo "  log:    $LOG_FILE"
  echo
done

echo "[DONE] All scenes processed."