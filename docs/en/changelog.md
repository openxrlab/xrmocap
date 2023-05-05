# Changelog

### v0.8.0 (05/05/2023/)

**Highlights**

- Refactor evaluation for MvP, mvpose_tracking, mvpose and fourdag, sharing the same super-class.
- Add smpl visualization and unit test, based on `minimal_pytorch_rasterizer`. Multi-person and multi-gender are supported.
- Add mmdeploy for faster human perception.

**New Features**

- Add `PriorConstraint` optimizer for 3D keypoints, filtering out poorly quality bboxes and limbs.
- Add mask in smpl_data. The person whose mask is zero will not be plotted.
- Add function `auto_load_smpl_data`, it chooses a correct class when you forget of which type the npz file is.
- Add class Timer for recording average time consumption.

**Refactors**

- Refactor evaluation metrics including MPJPE, PA-MPJPE, PCK, PCP, mAP, and recall.

### v0.7.0 (23/12/2022/)

**Highlights**

- Add [mview_mperson_end2end_estimator](https://github.com/openxrlab/xrmocap/blob/main/xrmocap/core/estimation/mview_mperson_end2end_estimator.py) for learning-based method.
- Add SMPLX support and allow smpl_data initiation in [mview_sperson_smpl_estimator](https://github.com/openxrlab/xrmocap/blob/main/xrmocap/core/estimation/mview_sperson_smpl_estimator.py).
- Add multiple optimizers, detailed joint weights and priors, grad clipping for better SMPLify results.
- Add [mediapipe_estimator](https://github.com/openxrlab/xrmocap/blob/main/xrmocap/human_perception/keypoints_estimation/mediapipe_estimator.py) for human keypoints2d perception.

**New Features**

- Add `mview_mperson_end2end_estimator`, performing MvP estimation on customized data.
- Add `mediapipe_estimator`, another alternative human keypoints2d perception method like `mmpose_top_down_estimator`.
- Add `RemoveDuplicate`  keypoints3d optimizer to remove duplicate MvP keypoints3d predictions.

**Refactors**

- Refactor `mview_sperson_smpl_estimator`, compatible with SMPLX.
- Refactor `SMPLify`, add grad clipping, joint angle priors, loss-parameter mapping, per-parameter optimizers, and body part weights.
- Refactor evaluation for learning-based methods.

### v0.6.0 (14/10/2022/)

**Highlights**

- Add [4D Association Graph](http://www.liuyebin.com/4dassociation/), the first Python implementation to reproduce this algorithm
- Add Multi-view multi-person top-down smpl estimation
- Add reprojection error point selector

**New Features**

- Add [4D Association Graph](http://www.liuyebin.com/4dassociation/), the first Python implementation to reproduce this algorithm
- Add Multi-view multi-person top-down smpl estimation
- Add structures for mview mperson kps3d/smpl estimator
- Add reprojection error point selector

**Refactors**

- Refactor Deformable and ProjAttn for MvP

### v0.5.0 (01/09/2022/)

**Highlights**

- Support [HuMMan Mocap](https://caizhongang.github.io/projects/HuMMan/) toolchain for multi-view single person SMPL estimation
- Reproduce [MvP](https://arxiv.org/pdf/2111.04076.pdf), a deep-learning-based SOTA for multi-view multi-human 3D pose estimation
- Reproduce [MVPose (single frame)](https://arxiv.org/abs/1901.04111) and [MVPose (temporal tracking and filtering)](https://ieeexplore.ieee.org/document/9492024), two optimization-based methods for multi-view multi-human 3D pose estimation
- Support SMPLify, SMPLifyX, SMPLifyD and SMPLifyXD

**New Features**

- Add peception module based on mmdet, mmpose and mmtrack
- Add [Shape-aware 3D Pose Optimization](https://ait.ethz.ch/projects/2021/multi-human-pose/)
- Add Keypoints3d optimizer and multi-view single-person api
- Add data_converter and data_visualization for shelf, campus and cmu panoptic datasets
- Add multiple selectors to support more point selection strategies for triangulation
- Add Keypoints and Limbs data structure
- Add multi-way matching registry
- Refactor the pictorial block (c/c++) in python
