# Multi-view Multi-person Evaluation

[TOC]

## Overview

This tool takes calibrated camera parameters, RGB sequences, 2d perception data and 3d ground-truth from `MviewMpersonDataset` as input, generate multi-view multi-person keypoints3d and evaluate on the Campus/Shelf/CMU-Panoptic datasets.

## Argument

- **evaluation_config**:
`evaluation_config` is the path to a `TopDownAssociationEvaluation` config file. For more details, see docs for `TopDownAssociationEvaluation` and the docstring in [code](../../../xrmocap/core/evaluation/top_down_association_evaluation.py).

Also, you can find our prepared config files at `configs/mvpose/*/eval_keypoints3d.py` or `configs/mvpose_tracking/*/eval_keypoints3d.py`.

## Example

Run the tool without tracking.

```bash
python tools/mview_mperson_evaluation.py \
      --evaluation_config configs/mvpose/shelf_config/eval_keypoints3d.py
```

Run the tool with tracking.

```bash
python tools/mview_mperson_evaluation.py \
      --evaluation_config configs/mvpose_tracking/shelf_config/eval_keypoints3d.py
```
