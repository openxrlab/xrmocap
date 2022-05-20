#!/usr/bin/env bash

set -x # print config to screen.

EXP_NAME="mvpose_kp17"
keypoints_number=17
############ shelf ############
START_FRAME=300
END_FRAME=302
DATASET_NAME="shelf"
enable_camera_id='0_1_2_3_4'
INPUT_ROOT="/home/coder/data"
############ campus ###########
# 350-470, 650-750
# START_FRAME=350
# END_FRAME=470
# DATASET_NAME="campus"
# enable_camera_id='0_1_2'
# INPUT_ROOT="/home/coder/data"
############ panoptic #########
# START_FRAME=200
# END_FRAME=202
# DATASET_NAME="panoptic_pizza"
# enable_camera_id='0_1_2_3_4'
# INPUT_ROOT="/home/coder/data/panoptic/"

affinity_type='geometry_mean'
affinity_reg_config='./config/affinity_estimation/resnet50_affinity_estimator.py'
affinity_reg_checkpoint='./weight/resnet50_reid_camstyle.pth.tar'


OUTPUT_DIR="/home/coder/output/mvpose/"

python -u tool/estimate_keypoints3d.py \
    -s ${START_FRAME} \
    -e ${END_FRAME} \
    --keypoints_number ${keypoints_number}\
    --input_root ${INPUT_ROOT}\
    --output_dir ${OUTPUT_DIR}${DATASET_NAME}\
    --dataset_name ${DATASET_NAME} \
    --enable_camera_id ${enable_camera_id} \
    --affinity_type ${affinity_type} \
    --affinity_reg_config ${affinity_reg_config} \
    --affinity_reg_checkpoint ${affinity_reg_checkpoint} \
    --exp_name ${EXP_NAME} \
    --use_dual_stochastic_SVT \
    --use_hybrid \
    --show
    # --use_tracking \
    # --show
