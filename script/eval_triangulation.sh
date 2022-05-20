#!/usr/bin/env bash
set -x # print config to screen.

DATA_ROOT="/home/coder/data"
RESULT_ROOT="/home/coder/output/mvpose/" # folder containing results

DATA="shelf" # shelf, campus, panoptic_ian
START_FRAME=300
END_FRAME=600
DATA_TYPE="coco" # kp17

EXP_NAME="mvpose_kp17"

python -u xrmocap/core/evaluation/evaluate_keypoints3d.py \
-s ${START_FRAME} \
-e ${END_FRAME} \
-d "${DATA}_${DATA_TYPE}" \
--exp_name ${EXP_NAME} \
--input_path ${DATA_ROOT} \
--result_path ${RESULT_ROOT}
