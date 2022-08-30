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
| mvp_campus.py | 96.7 | [model](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_campus.pth.tar) |


### Shelf

MvP for Shelf fine-tuned from the model weights pre-trained with 5 selected views in CMU Panoptic dataset is provided. The 5 selected views, HD camera view 3, 6, 12, 13 and 23 are the same views used in VoxelPose.

| Config | Shelf  | Download |
|:------:|:-------:|:--------:|
| mvp_shelf.py  | 97.1 | [model](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_shelf.pth.tar)  |


### CMU Panoptic

MvP for CMU Panoptic trained from stcratch with pre-trained Pose ResNet50 backbone is provided. The provided model weights were trained and evaluated with the 5 selected views same as VoxelPose (HD camera view 3, 6, 12, 13, 23).  A checkpoint trained with 3 selected views (HD camera view 3, 12, 23) is also provided as the pre-trained model weights for Campus dataset fine-tuning.

| Config | AP25 | AP100 | Recall@500 | MPJPE(mm) |Download |
|:------:|:----:|:----:|:---------:|:--------:|:--------:|
| mvp_panoptic.py | 91.5 | 97.9 | 99.85 |16.45 | [model](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_panoptic_5view.pth.tar) |
| mvp_panoptic_3cam.py | 54.7 | 95.1 | 98.83 |30.55 | [model](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_panoptic_3view_3_12_23.pth.tar)  |

### Pose ResNet50 backbone

All the checkpoints provided above was trained on top of the pre-trained [Pose ResNet50](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_pose_resnet50_panoptic.pth.tar) backbone weights.
