#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
JOBS=10   # ✅ 并行 worker 数（你要几个就填几，建议 2~4 先试）

LIST=/nas/qirui/only_in_train_val.txt
MASKS_ROOT=/nas/qirui/sam3/scenefun3d_ex/experiments4sam3_mask/all_scenes
DATA_ROOT=/nas/qirui/scenefun3d/val
OUT_DIR=/nas/qirui/sam3/scenefun3d_ex/Experiments4SceneGraph

mkdir -p "$OUT_DIR/logs"

CLEAN_LIST="$OUT_DIR/scene_ids.clean.txt"
grep -v '^[[:space:]]*$' "$LIST" | sed 's/[[:space:]]//g' > "$CLEAN_LIST"

export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "[GPU $GPU_ID] parallel run. JOBS=$JOBS OUT_DIR=$OUT_DIR"
echo "[GPU $GPU_ID] scene list: $CLEAN_LIST"

run_one () {
  local SID="$1"
  local SCENE_OUT_DIR="$OUT_DIR/$SID"
  mkdir -p "$SCENE_OUT_DIR"

  if [[ -f "$SCENE_OUT_DIR/scene_graph.json" ]]; then
    echo "[GPU $GPU_ID] skip $SID (scene_graph.json exists)"
    return 0
  fi

  echo "[GPU $GPU_ID] start $SID"

  # ✅ 限制每个 worker 的 CPU 线程数，避免“看起来很多线程”
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4

  python /nas/qirui/sam3/S2FUN/build_scene_graph.py \
    --scene_id "$SID" \
    --masks_root "$MASKS_ROOT/$SID/masks" \
    --data_root "$DATA_ROOT" \
    --depth_scale 1000 \
    --out_root "$OUT_DIR" \
    --save_ply \
    --use_depth_consistency \
    --min_vis 5 \
    --min_vis_frac 0 \
    --min_fg 2 \
    --aff_min_vis 5 \
    --aff_min_fg 2 \
    --aff_min_fg_ratio 0.05 \
    --aff_min_vis_frac 0.02 \
    --min_fg_ratio 0.3 \
    --use_dbscan_position \
    --dbscan_eps 1 \
    --aff_dbscan_eps 0.03 \
    --dbscan_min_samples 10 \
    --voxel_size 0 \
    --merge_nodes \
    --merge_min_common 50 \
    --merge_min_ratio 0.3 \
    > "$OUT_DIR/logs/${SID}.log" 2>&1

  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[GPU $GPU_ID] FAIL $SID (rc=$rc) log=$OUT_DIR/logs/${SID}.log"
    return $rc
  else
    echo "[GPU $GPU_ID] done $SID"
  fi
}

export -f run_one
export GPU_ID MASKS_ROOT DATA_ROOT OUT_DIR

cat "$CLEAN_LIST" | xargs -n1 -P"$JOBS" -I{} bash -lc 'run_one "$@"' _ {}

echo "ALL DONE (GPU $GPU_ID, JOBS=$JOBS)"