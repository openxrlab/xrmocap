#!/usr/bin/env bash

set -x

# CFG_FILE="configs/mvp/shelf_config/mvp_shelf.py"
# CFG_FILE="configs/mvp/campus_config/mvp_campus.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic.py"
# CFG_FILE="configs/mvp/panoptic_config/mvp_panoptic_3cam.py"

# Trained with xrmocap from scratch
# MODEL_PATH="weight/xrmocap_mvp_shelf.pth.tar"
# MODEL_PATH="weight/xrmocap_mvp_campus.pth.tar"
# MODEL_PATH="weight/xrmocap_mvp_panoptic_5view.pth.tar"
# MODEL_PATH="weight/xrmocap_mvp_panoptic_3view_3_12_23.pth.tar"


GPUS_PER_NODE=$1
CFG_FILE=$2
MODEL_PATH=$3


python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port 65530 \
    --use_env tools/eval_model.py \
    --cfg ${CFG_FILE} \
    --model_path ${MODEL_PATH}
