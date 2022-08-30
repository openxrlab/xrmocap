# Tool prepare_dataset

- [Overview](#overview)
- [Argument: converter_config](#argument-converter_config)
- [Argument: overwrite](#argument-overwrite)
- [Argument: disable_log_file](#argument-disable_log_file)
- [Argument: paths](#argument-paths)
- [Example](#example)

### Overview

This tool converts original dataset to our unified meta-data, with data converters controlled by configs.

### Argument: converter_config

`converter_config` is the path to a data_converter config file like below. If 2D perception data is not required by your method, set `bbox_detector` and `kps2d_estimator` to None. It will skip 2D perception and saves your time. For more details, see the docstring in [code](../../../xrmocap/data/data_converter/base_data_converter.py).

```python
type = 'ShelfDataCovnerter'
data_root = 'datasets/Shelf'
bbox_detector = dict(
    type='MMtrackDetector',
    mmtrack_kwargs=dict(
        config='config/human_detection/' +
        'mmtrack_deepsort_faster-rcnn_fpn_4e_mot17-private-half.py',
        device='cuda'))
kps2d_estimator = dict(
    type='MMposeTopDownEstimator',
    mmpose_kwargs=dict(
        checkpoint='weight/hrnet_w48_coco_wholebody' +
        '_384x288_dark-f5726563_20200918.pth',
        config='config/human_detection/mmpose_hrnet_w48_' +
        'coco_wholebody_384x288_dark_plus.py',
        device='cuda'))
scene_range = [[300, 600]]
meta_path = 'datasets/Shelf/xrmocap_meta_testset'
visualize = True
```

Also, you can find our prepared config files in `config/data/data_converter`, with or without perception.

### Argument: overwrite

By default, overwrite is False and there is a folder found at `meta_path`, the tool will raise an error, to avoid removal of existed files. Add `--overwrite` makes it True and allows the tool to overwrite any file below `meta_path`.

### Argument: disable_log_file

By default, disable_log_file is False and a log file named `converter_log_{time_str}.txt` will be written. Add `--disable_log_file` makes it True and the tool will only print log to console.

After the tool succeeds, you will find log file in  `meta_path`, otherwise it will be in `logs/`.

### Argument: paths

By default, `data_root` and `meta_path` are empty, the tool takes paths in converter config file. If both of them are set, the tool takes paths from argv.

### Examples

Run the tool when paths configured in `campus_data_converter_testset.py`.

```bash
python tool/prepare_dataset.py \
	--converter_config config/data/data_converter/campus_data_converter_testset.py
```

Run the tool with explicit paths.

```bash
python tool/prepare_dataset.py \
  --converter_config config/data/data_converter/campus_data_converter_testset.py \
  --data_root datasets/Campus \
  --meta_path datasets/Campus/xrmocap_meta_testset
```
