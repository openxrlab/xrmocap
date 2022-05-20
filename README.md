# XRMocap

## Installation

 - Compile the pictorial function for acceleration
```
sh script/download_test_data.sh
cd xrmocap/keypoints3d_estimation/lib/
python setup.py build_ext --inplace
```

## Prepare models and datasets

 -  **Prepare models**:
You can find CamStyle model in `weight` file

 - **Prepare the datasets**:
Download Shelf, Campus or CMU Panoptic, and put datasets to `./data/`

## Demo and Evaluate

### Run the demo
```
sh script/triangulation.sh
```

### Evaluate on the Shelf/Campus/CMU Panoptic datasets
```
sh script/eval_triangulation.sh
```
