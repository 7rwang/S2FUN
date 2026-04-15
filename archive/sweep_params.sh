#!/usr/bin/env bash
set -euo pipefail

# ==============================
# 配置区（自己改）
# ==============================
SCENE_ID=421657

MASKS_ROOT="/nas/qirui/sam3/scenefun3d_ex/anchor_cxt_csv/${SCENE_ID}/masks"
DATA_ROOT="/nas/qirui/scenefun3d/data"

BUILD_PY="/nas/qirui/sam3/S2FUN/build_scene_graph.py"
EVAL_PY="/nas/qirui/sam3/S2FUN/eval.py"

WORK_ROOT="/nas/qirui/sam3/S2FUN/dbscan_sweep/${SCENE_ID}"
mkdir -p "${WORK_ROOT}"

DBSCAN_EPS_LIST=(0.10 0.15 0.20 0.25 0.30)
DBSCAN_MIN_SAMPLES_LIST=(2 3 5 8 10)

# ==============================
# 清空总结果
# ==============================
ALL_JSON="${WORK_ROOT}/all_results.json"
echo "[]" > "${ALL_JSON}"

# ==============================
# 开始 sweep
# ==============================
for eps in "${DBSCAN_EPS_LIST[@]}"; do
  for ms in "${DBSCAN_MIN_SAMPLES_LIST[@]}"; do

    RUN_ID="eps_${eps}_ms_${ms}"
    RUN_DIR="${WORK_ROOT}/${RUN_ID}"
    mkdir -p "${RUN_DIR}"

    echo "==============================="
    echo "Running: ${RUN_ID}"
    echo "==============================="

    OUT_JSON="${RUN_DIR}/scene_graph.json"
    IDX_JSON="${RUN_DIR}/scene_graph_point_indices.json"
    EVAL_JSON="${RUN_DIR}/eval.json"
    SUMMARY_JSON="${RUN_DIR}/summary.json"

    # -------------------------
    # 1️⃣ build scene graph
    # -------------------------
    python "${BUILD_PY}" \
      --scene_id "${SCENE_ID}" \
      --masks_root "${MASKS_ROOT}" \
      --data_root "${DATA_ROOT}" \
      --out_json "${OUT_JSON}" \
      --use_depth_consistency \
      --use_dbscan_position \
      --dbscan_eps "${eps}" \
      --dbscan_min_samples "${ms}" \
      --min_vis 10 \
      --min_fg 2 \
      --min_fg_ratio 0.3 \
      --merge_nodes \
      --merge_min_common 50 \
      --merge_min_ratio 0.3 \
      > "${RUN_DIR}/build.log" 2>&1

    # -------------------------
    # 2️⃣ eval
    # -------------------------
    python "${EVAL_PY}" \
      --index-json "${IDX_JSON}" \
      --data-root "${DATA_ROOT}" \
      --scene-id "${SCENE_ID}" \
      --oracle-best-node \
      --out-json "${EVAL_JSON}" \
      > "${RUN_DIR}/eval.log" 2>&1

    # -------------------------
    # 3️⃣ 把指标抽出来写 summary.json
    # -------------------------
    python - <<PY
import json, os

eval_json = "${EVAL_JSON}"
summary_json = "${SUMMARY_JSON}"
eps = ${eps}
ms = ${ms}

if not os.path.exists(eval_json):
    data = {"status": "fail"}
else:
    obj = json.load(open(eval_json))
    m = obj.get("metrics", {})
    data = {
        "status": "ok",
        "dbscan_eps": eps,
        "dbscan_min_samples": ms,
        "mAP": m.get("mAP"),
        "AP50": m.get("AP50"),
        "AP25": m.get("AP25"),
        "mAR": m.get("mAR"),
        "AR50": m.get("AR50"),
        "AR25": m.get("AR25"),
    }

json.dump(data, open(summary_json, "w"), indent=2)
print("Saved summary:", summary_json)
PY

    # -------------------------
    # 4️⃣ 写入总 JSON
    # -------------------------
    python - <<PY
import json

all_file = "${ALL_JSON}"
summary_file = "${SUMMARY_JSON}"

all_data = json.load(open(all_file))
summary = json.load(open(summary_file))

all_data.append(summary)
json.dump(all_data, open(all_file, "w"), indent=2)
PY

  done
done

echo
echo "==============================="
echo "Sweep Done."
echo "All results saved to:"
echo "${ALL_JSON}"
echo "==============================="