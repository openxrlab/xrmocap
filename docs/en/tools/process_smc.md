# Tool process_smc

- [Overview](#overview)
- [Argument: estimator_config](#argument-estimator_config)
- [Argument: output_dir](#argument-output_dir)
- [Argument: disable_log_file](#argument-disable_log_file)
- [Argument: visualize](#argument-visualize)
- [Example](#example)

### Overview

This tool takes calibrated camera parameters and RGB frames from a SenseMoCap file as input, generate multi-view keypoints2d, keypoints3d and SMPLData.

### Argument: estimator_config

`estimator_config` is the path to a `MultiViewSinglePersonSMPLEstimator` config file. For more details, see docs for `MultiViewSinglePersonSMPLEstimator` and the docstring in [code](../../../xrmocap/estimation/mview_sperson_smpl_estimator.py).

Also, you can find our prepared config files at `config/estimation/mview_sperson_smpl_estimator.py`.

### Argument: output_dir

`output_dir` is the path to the directory saving all possible output files, including multi-view keypoints2d, keypoints3d and SMPLData, log and visualization videos.

### Argument: disable_log_file

By default, disable_log_file is False and a log file named `{smc_file_name}_{time_str}.txt` will be written. Add `--disable_log_file` makes it True and the tool will only print log to console.

### Argument: visualize

By default, visualize is False. Add `--visualize` makes it True and the tool will visualize keypoints3d with an orbit camera, overlay projected keypoints3d on some views, and overlay SMPL meshes on one view.

### Example

Run the tool with visualization.

```bash
python tools/process_smc.py \
	--estimator_config configs/humman_mocap/mview_sperson_smpl_estimator.py \
	--smc_path xrmocap_data/humman/raw_smc/p000105_a000195.smc \
	--output_dir xrmocap_data/humman/p000105_a000195_output \
	--visualize
```
