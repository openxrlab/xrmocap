# MVPose (Single frame)

- [Introduction](#introduction)
- [Prepare models and datasets](#prepare-models-and-datasets)
- [Results and Models](#results-and-models)

## Introduction

We provide the config files for MVPose (Single frame): [Fast and robust multi-person 3d pose estimation from multiple views](https://zju3dv.github.io/mvpose/).

[Official Implementation](https://github.com/zju3dv/mvpose)

```BibTeX
@inproceedings{dong2019fast,
  title={Fast and robust multi-person 3d pose estimation from multiple views},
  author={Dong, Junting and Jiang, Wen and Huang, Qixing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7792--7801},
  year={2019}
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

We evaluate MVPose (Single frame) on 3 popular benchmarks, and you can find the recommended configs in `configs/mvpose/*/eval_keypoints3d.py`.

`__bbox_thr__` is the threshold of bbox2d, you can set it higher to ignore incorrect 2D perception data, and we recommen setting it to 0.9. `n_cam_min` is the minimum amount of keypoints2d to triangulate, and the default value is 2. To improve the robustness of the algorithm, we set it to 2 on Campus and 3 on Shelf/CMU Panoptic.

### Campus

[Explain specific setting, metrics and other details]

| Config | Accuracy (PCP)  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./campus_config/eval_keypoints3d.py) | 93.69 | [model]() &#124; [log]() |


### Shelf

[Explain specific setting, metrics and other details]

| Config | Accuracy (PCP)  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./shelf_config/eval_keypoints3d.py) | - | [model]() &#124; [log]() |


### CMU Panoptic

[Explain view selection, training details etc.]

| Config | AP25 | MPJPE(mm) | Download |
|:------:|:----:|:---------:|:--------:|
| [eval_keypoints3d.py](./panoptic_config/eval_keypoints3d.py) | - | - | [model]() &#124; [log]() |
