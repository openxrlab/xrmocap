#!/usr/bin/env bash

python tools/mview_mperson_end2end_estimator.py \
    --output_dir ./output/estimation \
    --model_dir weight/xrmocap_mvp_shelf-22d1b5ed_20220831.pth \
    --estimator_config configs/modules/core/estimation/mview_mperson_end2end_estimator.py \
    --image_and_camera_param ./xrmocap_data/Shelf_50/image_and_camera_param.txt \
    --start_frame 300 \
    --end_frame 351  \
    --enable_log_file
