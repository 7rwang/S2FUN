#!/bin/bash

SERVER="pose_sr@a40"
REMOTE_BASE="/nas/qirui/scenefun3d/data"
LOCAL_BASE="./downloaded"

declare -a LIST=(
"421254 42444755"
"421393 42444924"
"421657 42445639"
"422356 42446579"
"422842 42897547"
"423070 42447202"
"423452 42897422"
"423957 42898340"
"434888 42899184"
)

for item in "${LIST[@]}"; do
    visit_id=$(echo $item | awk '{print $1}')
    video_id=$(echo $item | awk '{print $2}')

    echo "Downloading $visit_id/$video_id ..."

    rsync -avz \
    ${SERVER}:${REMOTE_BASE}/${visit_id}/${video_id}/hires_wide \
    ${LOCAL_BASE}/${visit_id}_${video_id}

done