# Tool visualize_dataset

- [Overview](#overview)
- [Argument: vis_config](#argument-vis_config)
- [Argument: overwrite](#argument-overwrite)
- [Argument: disable_log_file](#argument-disable_log_file)
- [Argument: paths](#argument-paths)
- [Example](#example)

### Overview

This tool loads our converted meta-data, visualize meta-data with background frames from original dataset, scene by scene.

### Argument: vis_config

`vis_config` is the path to a data_visualization config file like below. Visualization for perception 2D and groudtruth 3D is optional, controlled by `vis_percep2d` and `vis_gt_kps3d`. Visualization for predicted 3D will be done if `pred_kps3d_paths` is not empty, and each element is path to a keypoints3d npz file. For more details, see the docstring in [code](../../../xrmocap/data/data_visualization/base_data_visualization.py).

```python
type = 'MviewMpersonDataVisualization'
data_root = 'Shelf'
meta_path = 'datasets/Shelf/xrmocap_meta_testset'
output_dir = 'datasets/Shelf/xrmocap_meta_testset_visualization'
pred_kps3d_paths = ['datasets/Shelf/xrmocap_meta_testset/predicted_keypoints3d.npz']
bbox_thr = 0.96
vis_percep2d = True
vis_gt_kps3d = True
```

Also, you can find our prepared config files in `config/data/data_visualization`.

### Argument: overwrite

By default, overwrite is False and there is a folder found at `output_dir`, the tool will raise an error, to avoid removal of existed files. Add `--overwrite` makes it True and allows the tool to overwrite any file below `output_dir`.

### Argument: disable_log_file

By default, disable_log_file is False and a log file named `visualization_log_{time_str}.txt` will be written. Add `--disable_log_file` makes it True and the tool will only print log to console.

After the tool succeeds, you will find log file in  `output_dir`, otherwise it will be in `logs/`.

### Argument: paths

By default, `data_root`, `meta_path` and `output_dir` are empty, the tool takes paths in data_visualization config file. If all of them are set, the tool takes paths from argv.

### Examples

Run the tool when paths configured in `shelf_data_visualization_testset.py`.

```bash
python tool/visualize_dataset.py \
	--converter_config config/data/data_visualization/shelf_data_visualization_testset.py
```

Run the tool with explicit paths.

```bash
python tool/prepare_dataset.py \
  --converter_config config/data/data_converter/shelf_data_visualization_testset.py \
  --data_root datasets/Shelf \
  --meta_path datasets/Shelf/xrmocap_meta_testset \
  --output_dir datasets/Shelf/xrmocap_meta_testset/visualization
```
