# Introduction

This file introduces the framework design and file structure of xrmocap.


## Framework

### Optimization-based framework

[framework for single person]

The construction pipeline starts with frame-by-frame 2D keypoint detection and manual camera estimation. Then triangulation and bundle adjustment are applied to optimize the camera parameters as well as the 3D keypoints. Finally we sequentially fit the SMPL model to 3D keypoints to get a motion sequence represented using joint angles and a root trajectory. The following figure shows our pipeline overview.

[framework for multiple person]

For multiple person, two challenges will be posed.
One is to find correspondence between different views
The other is solve person-person occlusion

From the figure above, it illustrates that two modules are adding, namely matching module and tracking module.

### Learning-based framework

describe the component of each module (as in the paper)

how to incorporate optimization and learning-based methods into one framework



## File structures

```text
.
├── Dockerfile                   # Dockerfile for quick start
├── README.md                    # README
├── README_CN.md                 # README in Chinese
├── configs                      # Recommended configuration files for tools and modules
├── docs                         # docs
├── requirements                 # pypi requirements
├── scripts                      # scripts for downloading data, training and evaluation
├── tests                        # unit tests
├── tools                        # utility tools
└── xrmocap
    ├── core
    │   ├── estimation           # multi-view single-person or multi-pserson SMPL estimator
    │   ├── evaluation           # evluation on datasets
    │   ├── hook                 # hooks to registry
    │   ├── train                # end-to-end model trainer
    │   └── visualization        # visualization functions for data structures
    ├── data
    │   ├── data_converter       # modules for dataset converting into XRMoCap annotation
    │   ├── data_visualization   # modules for dataset visualization
    │   ├── dataloader           # implementation of torch.utils.data.Dataloader
    │   └── dataset              # implementation of torch.utils.data.Dataset
    ├── data_structure           # data structure for single-person SMPL(X/XD), multi-person keypoints etc.
    ├── human_perception         # modules for human perception
    ├── io                       # functions for Input/Output
    ├── model                    # neural network modules
    │   ├── architecture         # high-level models for a specific task
    │   ├── body_model           # re-implementation of SMPL(X) body models
    │   ├── loss                 # loss functions
    │   ├── mvp                  # models for MVP
    │   └── registrant           # re-implementation of SMPLify(X)
    ├── ops                      # operators for multi-view MoCap
    │   ├── projection           # modules for projecting 3D points to 2D points
    │   ├── top_down_association # multi-view human association and tracking on top-down-detection data
    │   └── triangulation        # modules for triangulating 2D points to 3D points
    │       └── point_selection  # modules for selecting good 2D points before triangulation
    ├── transform                # functions and classes for data transform, e.g., bbox, image, keypoints3d
    ├── utils                    # utility functions for camera, geomotry computation and others
    └── version.py               # digital version of XRMocap

```


Usage of each module/folder
