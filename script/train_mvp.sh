#!/usr/bin/env bash

set -x

CFG_FILE="config/mvp/campus_config/mvp_campus.py"
# CFG_FILE="config/mvp/shelf_config/mvp_shelf.py"
# CFG_FILE="config/mvp/panoptic_config/mvp_panoptic.py"
# CFG_FILE="config/mvp/panoptic_config/mvp_panoptic_3cam.py"

GPUS_PER_NODE=$1

python -m torch.distributed.launch \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 65520 \
        --use_env tool/train_model.py \
        --cfg ${CFG_FILE} \
