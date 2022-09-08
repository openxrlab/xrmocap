# Multi-view Multi-person Evaluation

- [Overview](#overview)
- [Argument](#argument)
- [Example](#example)

## Overview

This tool takes calibrated camera parameters, RGB sequences, 2d perception data and 3d ground-truth from `MviewMpersonDataset` as input, generate multi-view multi-person keypoints3d and evaluate on the Campus/Shelf/CMU-Panoptic datasets.

## Argument

- **enable_log_file**
By default, enable_log_file is False and the tool will only print log to console. Add `--enable_log_file` makes it True and a log file named `{smc_file_name}_{time_str}.txt` will be written.

- **evaluation_config**:
`evaluation_config` is the path to a `TopDownAssociationEvaluation` config file. For more details, see docs for `TopDownAssociationEvaluation` and the docstring in [code](../../../xrmocap/core/evaluation/top_down_association_evaluation.py).

Also, you can find our prepared config files at `configs/mvpose/*/eval_keypoints3d.py` or `configs/mvpose_tracking/*/eval_keypoints3d.py`.

## Example

Evaluate on the Shelf dataset and run the tool without tracking.

```python
python tools/mview_mperson_evaluation.py \
      --enable_log_file \
      --evaluation_config configs/mvpose/shelf_config/eval_keypoints3d.py
```

Evaluate on the Shelf dataset and run the tool with tracking.

```python
python tools/mview_mperson_evaluation.py \
      --enable_log_file \
      --evaluation_config configs/mvpose_tracking/shelf_config/eval_keypoints3d.py
```
