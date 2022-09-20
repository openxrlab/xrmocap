# Changelog

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
