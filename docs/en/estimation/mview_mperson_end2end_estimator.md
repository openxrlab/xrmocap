# Multi-view Multi-person End-to-end Estimator

- [Overview](#overview)
- [Arguments](#arguments)
- [Run](#run)
  - [Step0: estimate keypoints3d](#step0-estimate-keypoints3d)
  - [Step1: optimize keypoints3d](#step1-optimize-keypoints3d)
  - [Step2: estimate smpl](#step2-estimate-smpl)
- [Example](#example)

## Overview

This end-to-end estimator tool takes multi-view RGB sequences and multi-view calibrated camera parameters as input. By a simple call of `run()`, it outputs keypoints3d predicted by the model trained with learning based method in an end-to-end manner, as well as SMPL parameters for the multi-person in the multi-view scene.

## Arguments

- **output_dir**:
`output_dir` is the path to the directory saving all possible output files, including keypoints3d, SMPLData and visualization videos.
- **model_dir**:
`model_dir` is the path of the pretrained model for keypoints3d inference.

- **estimator_config**:
`estimator_config` is the path to a `MultiViewMultiPersonEnd2EndEstimator` config file, where `kps3d_model` configuration is necessary. `kps3d_optimizers` is a list of kps3d_optimizer, defined in `xrmocap/transform/keypoints3d/optim`. When inferring images stored on disk, set `load_batch_size` to a reasonable value will prevent your machine from out of memory, for MvP only `batch_size=1` is supported. For more details, see [config](../../../configs/modules/core/estimation/mview_mperson_end2end_estimator.py) and the docstring in [code](../../../xrmocap/core/estimation/mview_mperson_end2end_estimator.py).

- **image_and_camera_param**:
`image_and_camera_param` is a text file contains the image path and the corresponding camera parameters. Line 0 is the image path of the first view, and line 1 is the corresponding camera parameter path. Line 2 is the image path of the second view, and line 3 is the corresponding camera parameter path, and so on.
```text
xrmocap_data/Shelf_50/Shelf/Camera0/
xrmocap_data/Shelf_50/xrmocap_meta_testset_small/scene_0/camera_parameters/fisheye_param_00.json
xrmocap_data/Shelf_50/Shelf/Camera1/
xrmocap_data/Shelf_50/xrmocap_meta_testset_small/scene_0/camera_parameters/fisheye_param_01.json
```

- **start_frame**:
`start_frame` is the index of the start frame.

- **end_frame**:
`end_frame` is the index of the end frame.

- **enable_log_file**
By default, enable_log_file is False and the tool will only print log to console. Add `--enable_log_file` makes it True and a log file named `{smc_file_name}_{time_str}.txt` will be written.

- **disable_visualization**
By default, disable_visualization is False and the tool will visualize keypoints3d and SMPLData with an orbit camera, overlay SMPL meshes on one view.

## Run

Inside `run()`, there are three major steps of estimation, and details of each step are shown below.



### Step0: estimate keypoints3d

In this step, we process the multi-view RGB images with a configured `image_pipeline` and prepare the calibrated camera parameters as the meta data. With input images, meta data and pretrained model prepared, keypoints3d can be predicted in an end-to-end manner.

For more information relevant to pretrained model preparation and model inference, please refer to the [evaluation tutorial](../tools/eval_model.md).

### Step1: optimize keypoints3d

In this step, we apply some post-processing optimizers to the predicted keypoints3d, such as removing duplicate keypoints3d, adding tracking identities, optimizing the trajectory and interpolation for the missing points.

### Step2: estimate smpl

In this step, we estimate SMPL parameters from keypoints3d. For details of smpl fitting, see [smplify doc](../../../docs/en/model/smplify.md).

## Example

```python
python tools/mview_mperson_end2end_estimator.py \
    --output_dir ./output/estimation \
    --model_dir weight/mvp/xrmocap_mvp_shelf-22d1b5ed_20220831.pth \
    --estimator_config configs/modules/core/estimation/mview_mperson_end2end_estimator.py \
    --image_and_camera_param ./xrmocap_data/Shelf_50/image_and_camera_param.txt \
    --start_frame 300 \
    --end_frame 350  \
    --enable_log_file
```
