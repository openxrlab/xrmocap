# XRMocap

## Installation

Please refer to [Installation.md](./docs/Installation.md).

## Prepare models and datasets

 -  **Prepare models**:
You can find CamStyle model in `weight` file

 - **Prepare the datasets**:
Download Shelf, Campus or CMU Panoptic, and put datasets to `./data/`

## Demo and Evaluate

### Run the demo
```
python tool/estimate_keypoints3d.py --config ./config/kps3d_estimation/shelf_config/estimate_kps3d.py
```
 - If you want to use tracing on the input sequence, you can set `use_kalman_tracking` to True in config file.
### Evaluate on the Shelf/Campus/CMU Panoptic datasets
```
python xrmocap/core/evaluation/evaluate_keypoints3d.py --config ./config/kps3d_estimation/eval_kps3d_estimation.py
```
