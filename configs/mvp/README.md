# MvP

## Introduction

We provide the config files for MvP: [Direct multi-view multi-person 3d pose estimation](https://arxiv.org/pdf/2111.04076.pdf).

[Official Implementation](https://github.com/sail-sg/mvp)

```BibTeX
@article{zhang2021direct,
  title={Direct multi-view multi-person 3d pose estimation},
  author={Zhang, Jianfeng and Cai, Yujun and Yan, Shuicheng and Feng, Jiashi and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={13153--13164},
  year={2021}
}
```

## Results and Models

We evaluate MvP on 3 popular benchmarks, report the Percentage of Correct Parts (PCP) on Shelf and Campus dataset, Mean Per Joint Position Error (MPJPE), mAP and recall on CMU Panoptic dataset.

### Campus

MvP for Campus fine-tuned from the model weights pre-trained with 3 selected views in CMU Panoptic dataset is provided. Fine-tuning with the model pre-train with CMU Panoptic HD camera view 3, 6, 12 gives the best final performance on Campus dataset.

| Config | Campus  | Download |
|:------:|:-------:|:--------:|
| [mvp_campus.py](./campus_config/mvp_campus.py) | 96.77 | [model](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_campus-e6093968_20220831.pth) |


### Shelf

MvP for Shelf fine-tuned from the model weights pre-trained with 5 selected views in CMU Panoptic dataset is provided. The 5 selected views, HD camera view 3, 6, 12, 13 and 23 are the same views used in VoxelPose.

| Config | Shelf  | Download |
|:------:|:-------:|:--------:|
| [mvp_shelf.py](./shelf_config/mvp_shelf.py)  | 97.07 | [model](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_shelf-22d1b5ed_20220831.pth)  |


### CMU Panoptic

MvP for CMU Panoptic trained from stcratch with pre-trained Pose ResNet50 backbone is provided. The provided model weights were trained and evaluated with the 5 selected views same as VoxelPose (HD camera view 3, 6, 12, 13, 23).  A checkpoint trained with 3 selected views (HD camera view 3, 12, 23) is also provided as the pre-trained model weights for Campus dataset fine-tuning.

| Config | AP25 | AP100 | Recall@500 | MPJPE(mm) |Download |
|:------:|:----:|:----:|:---------:|:--------:|:--------:|
| [mvp_panoptic.py](./panoptic_config/mvp_panoptic.py) | 91.49 | 97.91 | 99.85 |16.45 | [model](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_panoptic_5view-1b673cdf_20220831.pth) |
| [mvp_panoptic_3cam.py](./panoptic_config/mvp_panoptic_3cam.py) | 54.66 | 95.12 | 98.83 |30.55 | [model](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_panoptic_3view_3_12_23-4b391740_20220831.pth)  |

### Pose ResNet50 Backbone

All the checkpoints provided above were trained on top of the pre-trained [Pose ResNet50](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/mvp/xrmocap_pose_resnet50_panoptic-5a2e53c9_20220831.pth) backbone weights.
