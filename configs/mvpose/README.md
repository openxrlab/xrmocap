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
Download weight and run
```
sh scripts/download_weight.sh
```
You can find CamStyle model in `weight` file.

- **Prepare the datasets**:

Convert original dataset to our unified meta-data, with data converters controlled by configs,
you can find more details in [dataset_preparation.md](../../docs/en/dataset_preparation.md).

## Results and Models

We evaluate MVPose (Single frame) on 3 popular benchmarks.

You can find the recommended configs in `configs/mvpose/*/eval_keypoints3d.py`.

### Campus

[Explain specific setting, metrics and other details]

| Config | Campus  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./campus_config/eval_keypoints3d.py) | - | [model](../../weight/resnet50_reid_camstyle.pth.tar) &#124; [log]() |


### Shelf

[Explain specific setting, metrics and other details]

| Config | Shelf  | Download |
|:------:|:-------:|:--------:|
| [eval_keypoints3d.py](./shelf_config/eval_keypoints3d.py) | - | [model](../../weight/resnet50_reid_camstyle.pth.tar) &#124; [log]() |


### CMU Panoptic

[Explain view selection, training details etc.]

| Config | AP25 | MPJPE(mm) | Download |
|:------:|:----:|:---------:|:--------:|
| [eval_keypoints3d.py](./panoptic_config/eval_keypoints3d.py) | - | - | [model](../../weight/resnet50_reid_camstyle.pth.tar) &#124; [log]() |
