# XRMocap

<div align="left">

[![actions](https://github.com/openxrlab/xrmocap/workflows/build/badge.svg)](https://github.com/openxrlab/xrmocap/actions)
[![codecov](https://codecov.io/gh/openxrlab/xrmocap/branch/main/graph/badge.svg)](https://codecov.io/gh/openxrlab/xrmocap)
[![PyPI](https://img.shields.io/pypi/v/xrmocap)](https://pypi.org/project/xrmocap/)
[![LICENSE](https://img.shields.io/github/license/openxrlab/xrmocap.svg)](https://github.com/openxrlab/xrmocap/blob/main/LICENSE)

</div>

## Introduction

English | [简体中文](README_CN.md)

XRMoCap is an open-source PyTorch-based codebase for the use of multi-view motion capture. It is a part of the [OpenXRLab](https://openxrlab.com/) project.

If you are interested in single-view motion capture, please refer to [mmhuman3d](https://github.com/open-mmlab/mmhuman3d) for more details.

https://user-images.githubusercontent.com/26729379/187710195-ba4660ce-c736-4820-8450-104f82e5cc99.mp4

A detailed introduction can be found in [introduction.md](./docs/en/tutorials/introduction.md).


### Major Features

- **Support popular multi-view motion capture methods for single person and multiple people**

  XRMoCap reimplements SOTA multi-view motion capture methods, ranging from single person to multiple people. It supports an arbitrary number of calibrated cameras greater than 2, and provides effective strategies to automatically select cameras.

- **Support keypoint-based and parametric human model-based multi-view motion capture algorithms**

  XRMoCap supports two mainstream motion representations, keypoints3d and SMPL(-X) model, and provides tools for conversion and optimization between them.

- **Integrate optimization-based and learning-based methods into one modular framework**

  XRMoCap decomposes the framework into several components, based on which optimization-based and learning-based methods are integrated into one framework. Users can easily prototype a customized multi-view mocap pipeline by choosing different components in configs.


## Benchmark

More details can be found in [benchmark.md](docs/en/benchmark.md).

Supported methods:

<details open>
<summary>(click to collapse)</summary>

- [x] [SMPLify](https://smplify.is.tue.mpg.de/) (ECCV'2016)
- [x] [SMPLify-X](https://smpl-x.is.tue.mpg.de/) (CVPR'2019)
- [x] [MVPose (Single frame)](https://zju3dv.github.io/mvpose/) (CVPR'2019)
- [x] [MVPose (Temporal tracking and filtering)](https://zju3dv.github.io/mvpose/) (T-PAMI'2021)
- [x] [Shape-aware 3D Pose Optimization](https://ait.ethz.ch/projects/2021/multi-human-pose/) (ICCV'2019)
- [x] [MvP](https://arxiv.org/pdf/2111.04076.pdf) (NeurIPS'2021)
- [x] [HuMMan MoCap](https://caizhongang.github.io/projects/HuMMan/) (ECCV'2022)

</details>

Supported datasets:

<details open>
<summary>(click to collapse)</summary>

- [x] [Campus](https://campar.in.tum.de/Chair/MultiHumanPose) (CVPR'2014)
- [x] [Shelf](https://campar.in.tum.de/Chair/MultiHumanPose) (CVPR'2014)
- [x] [CMU Panoptic](http://domedb.perception.cs.cmu.edu/) (ICCV'2015)

</details>


## Getting Started

Please see [getting_started.md](docs/en/getting_started.md) for the basic usage of XRMoCap.

## License

This project is released under the [Apache 2.0 license](LICENSE). Some supported methods may carry [additional licenses](docs/en/additional_licenses.md).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{xrmocap,
    title={OpenXRLab Multi-view Motion Capture Toolbox and Benchmark},
    author={XRMoCap Contributors},
    howpublished = {\url{https://github.com/openxrlab/xrmocap}},
    year={2022}
}
```

## Contributing

We appreciate all contributions to improve XRMoCap. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

XRMoCap is an open source project that is contributed by researchers and engineers from both the academia and the industry.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenXRLab

- [XRPrimer](https://github.com/openxrlab/xrprimer): OpenXRLab foundational library for XR-related algorithms.
- [XRSLAM](https://github.com/openxrlab/xrslam): OpenXRLab Visual-inertial SLAM Toolbox and Benchmark.
- [XRSfM](https://github.com/openxrlab/xrsfm): OpenXRLab Structure-from-Motion Toolbox and Benchmark.
- [XRLocalization](https://github.com/openxrlab/xrlocalization): OpenXRLab Visual Localization Toolbox and Server.
- [XRMoCap](https://github.com/openxrlab/xrmocap): OpenXRLab Multi-view Motion Capture Toolbox and Benchmark.
- [XRMoGen](https://github.com/openxrlab/xrmogen): OpenXRLab Human Motion Generation Toolbox and Benchmark.
- [XRNeRF](https://github.com/openxrlab/xrnerf): OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark.
