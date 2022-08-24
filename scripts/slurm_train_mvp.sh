#!/usr/bin/env bash

set -x

CFG_FILE="configs/mvp/campus_config/mvp_campus.py"
# CFG_FILE="configs/mvp/shelf_config/mvp_shelf.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic_3cam.py"

PARTITION=$1
JOB_NAME=mvp_train
GPUS_PER_NODE=$2
CPUS_PER_TASK=1

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch \
          --nproc_per_node=${GPUS_PER_NODE} \
          --master_port 65521 \
          --use_env tool/train_model.py \
          --cfg ${CFG_FILE} \
