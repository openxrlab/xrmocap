# Multi-view Multi-person Top-down SMPL Estimator

- [Overview](#overview)
- [Arguments](#arguments)
- [Run](#run)
  - [Step0: estimate perception2d](#step0-estimate-keypoints2d)
  - [Step1: establish cross-frame and cross-person associations](#step1-establish-cross-frame-and-cross-person-associations)
  - [Step2: estimate keypoints3d](#step2-estimate-keypoints3d)
  - [Step3: estimate smpl](#step3-estimate-smpl)
- [Example](#example)

## Overview

This tool takes multi-view RGB sequences and multi-view calibrated camera parameters as input. By a simple call of `run()`, it outputs triangulated keypoints3d and SMPL parameters for the multi-person in the multi-view scene.

## Arguments

- **output_dir**:
`output_dir` is the path to the directory saving all possible output files, including keypoints3d, SMPLData and visualization videos.

- **estimator_config**:
`estimator_config` is the path to a `MultiViewMultiPersonTopDownEstimator` config file, where `bbox_detector`, `kps2d_estimator`, `associator`, `triangulator` and `smplify` are necessary. Every element of `point_selectors` are configs for point selector defined in `xrmocap/ops/triangulation/point_selection`. `kps3d_optimizers` is a list of kps3d_optimizer, defined in `xrmocap/transform/keypoints3d/optim`. When inferring images stored on disk, set `load_batch_size` to a reasonable value will prevent your machine from out of memory. For more details, see [config](../../../configs/modules/core/estimation/mview_mperson_topdown_estimator.py) and the docstring in [code](../../../xrmocap/core/estimation/mview_mperson_topdown_estimator.py).

- **image_and_camera_param**:
`image_and_camera_param` is a text file contains the image path and the corresponding camera parameters. Line 0 is the image path of the first view, and line 1 is the corresponding camera parameter path. Line 2 is the image path of the second view, and line 3 is the corresponding camera parameter path, and so on.

- **start_frame**:
`start_frame` is the index of the start frame.

- **end_frame**:
`end_frame` is the index of the end frame.

- **enable_log_file**
By default, enable_log_file is False and the tool will only print log to console. Add `--enable_log_file` makes it True and a log file named `{smc_file_name}_{time_str}.txt` will be written.

- **disable_visualization**
By default, disable_visualization is False and the tool will visualize keypoints3d and SMPLData with an orbit camera, overlay SMPL meshes on one view.

## Run

Inside `run()`, there are three major steps of estimation, and details of each step are shown in the diagram below.

### Step0: estimate perception2d

In this step, we perform a top-down keypoints2d estimation, detect bbox2d by `bbox_detector`, and detect keypoints2d in every bbox by `kps2d_estimator`. You can choose the model and weight you like by modifying the config file.

### Step1: establish cross-frame and cross-person associations
In this step, we match the keypoints2d across views by `associator` and add temporal tracking and filtering. For recommended configs on `associator`, you can check out the [README.md](../../../configs/mvpose_tracking/README.md)

### Step2: estimate keypoints3d

In this step, we split the estimation into three sub-steps: point selection, triangulation and optimization. Every sub-step can be skipped by passing `None` in config except triangulation. We use cascaded point selectors in `point_selectors` to select 2D points from well-calibrated views, for triangulation. After multi-view triangulation, in the third sub-step, we use cascaded keypoints3d optimizers in `kps3d_optimizers` to optimize keypoints3d.

### Step3: estimate smpl

In this step, we estimate SMPL parameters from keypoints3d. For details of smpl fitting, see [smplify doc](../../../docs/en/model/smplify.md).

## Example

```python
python tools/mview_mperson_topdown_estimator.py \
      --image_and_camera_param 'data/image_and_camera_param.txt' \
      --start_frame 0 \
      --end_frame 10 \
      --enable_log_file
```
