# Dataset preparation

[TOC]

### Overview

Our data pipeline converts original dataset to our unified meta-data, with data converters controlled by configs.

### Supported datasets

| Dataset name | Dataset page                                      | Download from public                                         | Download from OpenXRLab                                      |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Campus       |                                                   | [CampusSeq1.tar.bz2](https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/CampusSeq1.tar.bz2) | [CampusSeq1.tar.bz2](http://10.4.11.59:18080/resources/XRlab/dataset/CampusSeq1.tar.bz2) |
| Shelf        |                                                   | [Shelf.tar.bz2](https://www.campar.in.tum.de/public_datasets/2014_cvpr_belagiannis/Shelf.tar.bz2) | [Shelf.tar.bz2](http://10.4.11.59:18080/resources/XRlab/dataset/Shelf.tar.bz2) |
| CMU Panoptic | [home page](http://domedb.perception.cs.cmu.edu/) |                                                              |                                                              |

### Prepare a dataset

Edit config file for data_converter first, set Dataset type and path correctly. If 2D perception data is necessary for you method, set `bbox_detector` and `kps2d_estimator` like `config/data/data_converter/*_w_perception.py`.

```python
type = 'ShelfDataCovnerter'
data_root = 'datasets/Shelf'
bbox_detector = None
kps2d_estimator = None
scene_range = [[300, 600]]
meta_path = 'datasets/Shelf/xrmocap_meta_testset'
visualize = True
```

Run the dataset preparation tool as below, you will find meta-data at `meta_path`.

```bash
python tool/prepare_dataset.py --converter_config config/data/data_converter/campus_data_converter_testset.py
```

### Visualization

We also offer a tool for visualizing your predicted keypoints3d. To visualize predicted keypoints3d, ground truth keypoints3d, and perception 2d data whose b-box score is higher than 0.96, write a config like this.

```python
type = 'MviewMpersonDataVisualization'
data_root = 'Shelf'
output_dir = 'datasets/Shelf/visualization_output'
meta_path = 'datasets/Shelf/xrmocap_meta_testset'
pred_kps3d_paths = ['datasets/Shelf/xrmocap_meta_testset/predicted_keypoints3d.npz']
bbox_thr = 0.96
vis_percep2d = True
vis_gt_kps3d = True
```

Run the dataset visualization tool as below, you will find videos at `output_dir`.

```bash
python tool/visualize_dataset.py --vis_config config/data/data_visualization/shelf_data_visualization_testset.py
```
