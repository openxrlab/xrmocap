# Multi-view Single-person SMPL Estimator

- [Overview](#overview)
- [Arguments](#arguments)
- [Run](#run)
  - [Step0: estimate_keypoints2d](#step0-estimate_keypoints2d)
  - [Step1: estimate_keypoints3d](#step1-estimate_keypoints3d)
  - [Step2: estimate_smpl](#step2-estimate_smpl)
- [Example](#example)

## Overview

This tool takes multi-view videos and multi-view calibrated camera parameters as input. By a simple call of `run()`, it outputs detected keypoints2d, triangulated keypoints3d and SMPL parameters for the single person in the multi-view scene.

## Arguments

To construct an estimator instance of this class, you will need a config file like [config/estimation/mview_sperson_smpl_estimator.py](../../../config/estimation/mview_sperson_smpl_estimator.py). `work_dir` makes no sense in this estimator, could be any value, while `bbox_detector`, `kps2d_estimator`, `triangulator` and `smplify` are necessary. `triangulator` shall be a config for triangulator defined in `xrmocap/ops/triangulation`, instead of xrprimer triangulator. `smplify` can be a config for either `class SMPLify` or `class SMPLifyX`. `cam_pre_selector` , `cam_selector` and every list element of `final_selectors` are configs for point selector defined in `xrmocap/ops/triangulation/point_selection`. `kps3d_optimizers` is a list of kps3d_optimizer, defined in `xrmocap/transform/keypoints3d/optim`. When inferring images stored on disk, set `load_batch_size` to a reasonable value will prevent your machine from out of memory.

For more details, see the docstring in [code](../../../xrmocap/estimation/mview_sperson_smpl_estimator.py).

## Run

Inside `run()`, there are three major steps of estimation, and details of each step are shown in the diagram below.

### Step0: estimate keypoints2d

In this step, we perform a top-down keypoints2d estimation, detect bbox2d by `bbox_detector`, and detect keypoints2d in every bbox by `kps2d_estimator`. You can choose the model and weight you like by modifying the config file.

### Step1: estimate keypoints3d

In this step, we split the estimation into four sub-steps: camera selection, point selection, triangulation and optimization. Every sub-step can be skipped by passing `None` in config except triangulation. First, we use `cam_pre_selector` to select good 2D points from all detected keypoints2d, and select well-calibrated cameras by `cam_selector`. Second, we use cascaded point selectors in `final_selectors` to select 2D points from well-calibrated views, for triangulation. After multi-view triangulation, in the forth sub-step, we use cascaded keypoints3d optimizers in `kps3d_optimizers` to optimize keypoints3d, and the result of optimization will be the return value of step `estimate_keypoints3d`.

### Step2: estimate SMPL

In this step, we estimate SMPL or SMPLX parameters from keypoints3d of last step. For details of smpl fitting, see [smplify doc](../../../docs/en/model/smplify.md).

<img src="http://assets.processon.com/chart_image/62a05e9a5653bb0ca01eb161.png"/>

## Example

```bash
import numpy as np

from xrmocap.estimation.builder import build_estimator
from xrprimer.data_structure.camera import FisheyeCameraParameter


# multi-view camera parameter list
cam_param_list = [FisheyeCameraParameter.fromfile(
    f'fisheye_param_{idx:02d}.json') for idx in range(10)]
# multi-view image array
mview_img_arr = np.zeros(shape=(10, 150, 1080, 1920, 3), dtype=np.uint8)
# build an estimator
mview_sperson_smpl_estimator = build_estimator(estimator_config)

# run the estimator on image array
keypoints2d_list, keypoints3d, smpl_data = mview_sperson_smpl_estimator.run(
    cam_param=cam_param_list, img_arr=mview_img_arr)
```
