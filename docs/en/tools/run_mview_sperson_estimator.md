## Tool run_mview_sperson_estimator


- [Overview](#overview)
- [Argument: estimator_config](#argument-estimator_config)
- [Argument: output_dir](#argument-output_dir)
- [Argument: disable_log_file](#argument-disable_log_file)
- [Argument: visualize](#argument-visualize)
- [Example](#example)

### Overview

If you don't want to write code to call the `MultiViewSinglePersonSMPLEstimator` yourself, it's okay. We provide [this tool](../../../tools/run_mview_sperson_estimator.py) that can be run under a specific directory structure. Before using this tool, you need to organize the files required by the estimator according to the rules below.

```
your_dataset_root
└── xrmocap_meta
    ├── dataset_name.txt
    ├── scene_0
    │   ├── camera_parameters
    │   │   ├── fisheye_param_00.json
    │   │   ├── fisheye_param_01.json
    │   │   └── ...
    │   ├── image_list_view_00.txt
    │   ├── image_list_view_01.txt
    │   ├── ...
    │   ├── images_view_00
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   ├── images_view_01
    │   │   └── ...
    │   └──  ...
    └── scene_1
        └── ...
```
`fisheye_param_{view_idx}.json` is a json file for XRPrimer FisheyeCameraParameter, please refer to [XRPrimer docs](https://github.com/openxrlab/xrprimer/blob/main/docs/en/data_structure/camera.md) for details.  
`image_list_view_{view_idx}.txt` is a list of image paths relative to your dataset root, here's an example.
```
xrmocap_meta/scene_0/images_view_00/000000.jpg
xrmocap_meta/scene_0/images_view_00/000001.jpg
...
```

### Argument: estimator_config

`estimator_config` is the path to a `MultiViewSinglePersonSMPLEstimator` config file. For more details, see docs for `MultiViewSinglePersonSMPLEstimator` and the docstring in [code](../../../xrmocap/estimation/mview_sperson_smpl_estimator.py).

Also, you can find our prepared config files at `config/estimation/mview_sperson_smpl_estimator.py`.

### Argument: data_root

`data_root` is the path to the root directory of dataset. In the file tree example given above, `data_root` should point to `your_dataset_root`.

### Argument: meta_path

`meta_path` is the path to the meta data directory. In the file tree example given above, `meta_path` should point to `xrmocap_meta`.

### Argument: output_dir

`output_dir` is the path to the directory saving all possible output files, including multi-view keypoints2d, keypoints3d, SMPLData and visualization videos.

### Argument: disable_log_file

By default, disable_log_file is False and a log file named `{tool_name}_{time_str}.txt` will be written. Add `--disable_log_file` makes it True and the tool will only print log to console.

### Argument: visualize

By default, visualize is False. Add `--visualize` makes it True and the tool will visualize keypoints3d with an orbit camera, overlay projected keypoints3d on some views, and overlay SMPL meshes on one view.

### Example

Run the tool with visualization.

```bash
python tools/mview_sperson_estimator.py \
	--data_root xrmocap_data/humman_dataset \
	--meta_path xrmocap_data/humman_dataset/xrmocap_meta \
	--visualize
```
