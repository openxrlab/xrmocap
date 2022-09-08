#!/usr/bin/env bash

set -x

# CFG_FILE="configs/mvp/campus_config/mvp_campus.py"
# CFG_FILE="configs/mvp/shelf_config/mvp_shelf.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic_3cam.py"

GPUS_PER_NODE=$1
CFG_FILE=$2

python -m torch.distributed.launch \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 65530 \
        --use_env tools/train_model.py \
        --cfg ${CFG_FILE} \
