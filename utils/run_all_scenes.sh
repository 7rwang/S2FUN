#!/usr/bin/env bash
set -euo pipefail

CSV="/nas/qirui/sam3/scenefun3d_ex/src/parse_result_30scenes.csv"
DATA_ROOT="/nas/qirui/scenefun3d/val"
OUT_ROOT="/nas/qirui/sam3/scenefun3d_ex/Experiments4SceneGraph/all_scenes"
IMAGE_SUBDIR="hires_wide"
THR="0.5"
NPROC="8"
GPUS="1,2,3,4,5,6,7,8"   # 改成你机器真实的 8 张卡编号

RERUN_SCENES="
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
"

# 从 CSV 提取唯一 scene_id（跳过表头）
SCENES=${RERUN_SCENES}

for SID in ${SCENES}; do
  SCENE_DIR="${OUT_ROOT}/${SID}"
  DONE_FLAG="${SCENE_DIR}/DONE"

  if [[ -f "${DONE_FLAG}" ]]; then
    echo "[SKIP] scene ${SID} already done."
    continue
  fi

  echo "============================================================"
  echo "[RUN ] scene ${SID}"
  echo "============================================================"

  # 每个 scene 单独一个日志，便于排错
  mkdir -p "${SCENE_DIR}"
  LOG="${SCENE_DIR}/run.log"

  set +e
  CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} /nas/qirui/sam3/S2FUN/sam3_detection/scenefun3d_sam3_image.py \
    --csv-path "${CSV}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUT_ROOT}" \
    --scene-id "${SID}" \
    --image-subdir "${IMAGE_SUBDIR}" \
    --thr "${THR}" \
    --save-vis 2>&1 | tee "${LOG}"
  RET=${PIPESTATUS[0]}
  set -e

  if [[ ${RET} -eq 0 ]]; then
    date > "${DONE_FLAG}"
    echo "[DONE] scene ${SID}"
  else
    echo "[FAIL] scene ${SID} (exit=${RET}). See ${LOG}"
    # 不退出，继续跑后面的 scene（你要全量，这样更实际）
    continue
  fi
done
