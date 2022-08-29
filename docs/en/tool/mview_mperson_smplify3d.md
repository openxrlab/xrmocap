# Multi-view Multi-person SMPLify3D

[TOC]

## Overview

This tool could generate multi-view multi-person SMPLData from keypoints3d.

## Argument

- **image_dir**:
`image_dir` is the path to the directory containing RGB sequence. If you have five views of the RGB sequences, there should be five folders in `image_dir`.

- **start_frame**:
`start_frame` is the index of the start frame.

- **end_frame**:
`end_frame` is the index of the end frame.

- **bbox_thr**:
`bbox_thr` is the threshold of the 2d bbox, which should be the same as the threshold used to generate the keypoints3d.

- **keypoints3d_path**:
`keypoints3d_path` is the path to the keypoints3d file.

- **fisheye_param_dir**:
`fisheye_param_dir` is the path to the directory containing camera parameter.

- **perception2d_path**:
`perception2d_path` is the path to the 2d perception data.

- **matched_list_path**:
`matched_list_path` is the path to the matched keypoints2d index from different views.

- **output_dir**:
`output_dir` is the path to the directory saving all possible output files, including SMPLData and visualization videos.

- **estimator_config**:
`estimator_config` is the path to a `MultiViewMultiPersonSMPLEstimator` config file. For more details, see docs for `MultiViewMultiPersonSMPLEstimator` and the docstring in [code](../../../xrmocap/core/estimation/mview_mperson_smpl_estimator.py).

- **visualize**:
By default, visualize is False. Add `--visualize` makes it True and the tool will visualize SMPLData with an orbit camera, overlay SMPL meshes on one view.


## Example

Run the tool with visualization.

```bash
python tool/mview_mperson_smplify3d.py \
      --image_dir 'xrmocap_data/Shelf' \
      --start_frame 300 \
      --end_frame 600 \
      --keypoints3d_path 'output/Shelf/scene0_pred_keypoints3d.npz' \
      --fisheye_param_dir 'xrmocap_data/Shelf/xrmocap_meta_test/scene_0/camera_parameters' \
      --perception2d_path 'xrmocap_data/Shelf/xrmocap_meta_test/scene_0/perception_2d.npz' \
      --matched_list_path 'output/Shelf/scene0_matched_kps2d_idx.npy' \
      --output_dir 'output/Shelf' \
      --estimator_config 'configs/modules/core/estimation/mview_mperson_smpl_estimator.py' \
      --visualize
```
