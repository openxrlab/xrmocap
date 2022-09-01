# Add new Datasets

### Overview

This doc is a tutorial for how to support a new public dataset, or data collected by user.

### Online conversion

For online conversion, program does not write any file to disk. You have to define a new sub-class of `torch.utils.data.Dataset`, loading data from origin files, and return the same values in same sequence in `__getitem__()` as our [datasets](../../../xrmocap/data/dataset).

### Offline conversion (recommend)

For offline conversion, we convert the origin dataset into annotations in a unified format, save the annotations to disk, before training or evaluation starts. Such a conversion module is called `data_converter` in XRMoCap, and you can find examples in [xrmocap/data/data_converter](../../../xrmocap/data/data_converter).

#### File tree of our unified format

```
Dataset_xxx
├── (files in Dataset_xxx)
└── xrmocap_meta_xxxx
    ├── dataset_name.txt
    ├── scene_0
    │   ├── camera_parameters
    │   │   ├── fisheye_param_00.json
    │   │   ├── fisheye_param_01.json
    │   │   ├── ...
    │   │   └── fisheye_param_{n_view-1}.json
    │   ├── image_list_view_00.txt
    │   ├── image_list_view_01.txt
    │   ├── ...
    │   ├── image_list_view_{n_view-1}.txt
    │   ├── keypoints3d_GT.npz
    │   └── perception_2d.npz
    ├── scene_1
    │   └── ...
    └── scene_{n_scene-1}
        └── ...
```

#### Camera parameters of our unified format

Each scene has its independent multi-view camera parameters, and each json file is dumped by `class FisheyeCameraParameter` in [XRPrimer](https://github.com/openxrlab/xrprimer/blob/main/docs/en/data_structure/camera.md#fisheye).

#### Image list of our unified format

In a scene whose frame length is `n_frame`, number of cameras is `n_view`, there are `n_view` image list files, and every file has `n_frame` lines inside, take the `frame_idx`-th line in file `image_list_view_{view_idx}.txt`, we get a path of image relative to dataset_root(Dataset_xxx).

#### Keypoints3d groundtruth of our unified format

`keypoints3d_GT.npz` is a file dumped by `class Keypoints`, and it can be load by `keypoints3d = Keypoints.fromfile()`. In a scene whose frame length is `n_frame`, max number of objects is  `n_person`, number of single person's keypoints is `n_kps`, `keypoints3d.get_keypoints()` returns an ndarray in shape [n_frame, n_person, n_kps, 4], and `keypoints3d.get_mask()` is an ndarray in shape [n_frame, n_person, n_kps] which indicates which person and which keypoint is valid at a certain frame.

#### Perception 2D of our unified format

`perception_2d.npz` is an compressed npz file of a python dict, whose structure lies below:

```python
perception_2d_dict = dict(
	bbox_tracked=True,
  # True if bbox indexes have nothing to do with identity
  bbox_convention='xyxy',
  # xyxy or xywh
  kps2d_convention='coco',
  # or any other convention defined in KEYPOINTS_FACTORY
  bbox2d_view_00=bbox_arr_0,
  ...
  # an ndarray of bboxes detected in view 0, in shape (n_frame, n_person_max, 5)
  # bbox_arr[..., 4] are bbox scores
  # if bbox_arr[f_idx, p_idx, 4] == 0, bbox at bbox_arr[f_idx, p_idx, :4] is invalid
  kps2d_view_00=kps2d_arr_0,
  ...
  # an ndarray of keypoints2d detected in view 0, in shape (n_frame, n_person_max, n_kps, 3)
  # kps2d_arr[..., 2] are keypoints scores
  kps2d_mask_view_00=kps2d_mask_0,
  ...
  # a mask ndarray of keypoints2d in view 0, in shape (n_frame, n_person_max, n_kps)
  # if kps2d_mask[f_idx, p_idx, kps_idx] == 0, kps at kps2d_arr[f_idx, p_idx, kps_idx, :] is invalid
)
```

### Class data_converter

For frequent conversion, it's better to define a data_converter class inherited from `BaseDataCovnerter`. After that, you can use our prepare_dataset tool to convert the new dataset. See the [tool tutorial](../tool/prepare_dataset.md) for details.
