# MVPose (Temporal tracking and filtering)

- [Introduction](#introduction)
- [Prepare models and datasets](#prepare-models-and-datasets)
- [Results and Models](#results-and-models)

## Introduction

We provide the config files for MVPose (Temporal tracking and filtering): [Fast and robust multi-person 3d pose estimation and tracking from multiple views](https://zju3dv.github.io/mvpose/).

```BibTeX
@article{dong2021fast,
  title={Fast and robust multi-person 3d pose estimation and tracking from multiple views},
  author={Dong, Junting and Fang, Qi and Jiang, Wen and Yang, Yurou and Huang, Qixing and Bao, Hujun and Zhou, Xiaowei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## Prepare models and datasets

- **Prepare models**:

```
sh scripts/download_weight.sh
```
You can find `resnet50_reid_camstyle.pth.tar` in `weight` file.

- **Prepare the datasets**:

Convert original dataset to our unified meta-data, with data converters controlled by configs,
you can find more details in [dataset_preparation.md](../../docs/en/dataset_preparation.md).

## Results and Models

We evaluate MVPose (Temporal tracking and filtering) on 3 popular benchmarks, and you can find the recommended configs in `configs/mvpose_tracking/*/eval_keypoints3d.py`.

`interval` is the global matching interval, that is, the maximum number of frames for Kalman filtering. If the `interval` is set too large, the accuracy of the estimation will be degraded. We recommen within 50 frames. `__bbox_thr__` is the threshold of bbox2d, you can set it higher to ignore incorrect 2D perception data, and we recommen setting it to 0.9. `best_distance` is the threshold at which the current-frame keypoints2d successfully matches the last-frame keypoints2d, for the different dataset, it needs to be adjusted. `n_cam_min` is the minimum amount of keypoints2d to triangulate, and the default value is 2. To improve the robustness of the algorithm, we set it to 2 on Campus and 3 on Shelf/CMU Panoptic.

### Campus

| Config | Accuracy (PCP)  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./campus_config/eval_keypoints3d.py) | 93.95 | [model]() &#124; [log]() |


### Shelf

| Config | Accuracy (PCP)  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./shelf_config/eval_keypoints3d.py) | 96.5 | [model]() &#124; [log]() |


### CMU Panoptic

| Config | AP25 | MPJPE(mm) | Download |
|:------:|:----:|:---------:|:--------:|
| [eval_keypoints3d.py](./panoptic_config/eval_keypoints3d.py) | - | - | [model]() &#124; [log]() |
