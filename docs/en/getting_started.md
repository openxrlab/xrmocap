# Getting started

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Body Model Preparation (Optional)](#body-model-preparation-optional)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Training](#training)
- [More tutorials](#more-tutorials)

## Installation

Please refer to [installation.md](./installation.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](./dataset_preparation.md) for data preparation.

## Body Model Preparation (Optional)

If you want to obtain keypoints3d, the body model is not necessary.
If you want to infer SMPL as well, you can prepare the body_model as follows.

- [SMPL](https://smpl.is.tue.mpg.de/) v1.0 is used in our experiments.
  - Neutral model can be downloaded from [SMPLify](https://smplify.is.tue.mpg.de/).
  - All body models have to be renamed in `SMPL_{GENDER}.pkl` format. <br/>
    For example, `mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl SMPL_NEUTRAL.pkl`
- [smpl_mean_params.npz](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4)

- [gmm_08.pkl](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/gmm_08.pkl?versionId=CAEQQRiBgMCH6YWAmRgiIDE4NTI0MDQ5NDBjYjQzY2U4NDM5MzQxMWZjNjY5NzRj)

Download the above resources and arrange them in the following file structure:

```text
xrmocap
├── xrmocap
├── docs
├── tests
├── tools
├── configs
└── xrmocap_data
    └── body_models
        ├── gmm_08.pkl
        ├── smpl_mean_params.npz
        └── smpl
            ├── SMPL_FEMALE.pkl
            ├── SMPL_MALE.pkl
            └── SMPL_NEUTRAL.pkl
```

## Inference

We provide a demo script to estimate SMPL parameters for single-person or multi-person from multi-view synchronized input images or videos. With this demo script, you only need to choose a method, we currently support two types of methods, namely, optimization-based approaches and end-to-end learning algorithms, specify a few arguments, and then you can get the estimated results.

We assume that the cameras have been calibrated. If you want to know more about camera calibration, refer to [XRPrimer](https://github.com/openxrlab/xrprimer/blob/main/docs/en/tools/calibrate_multiple_cameras.md) for more details.


### Perception Model

Prepare perception models, including detection, 2d pose estimation, tracking and CamStyle models.

```
sh scripts/download_weight.sh
```
You could find perception models in `weight` file.

### Single Person

Currently, we only provide optimization-based method for single person estimation.

1. Download body model. Please refer to [Body Model Preparation](#body-model-preparation-optional)
2. Download a 7z file from [humman dataset](https://drive.google.com/drive/folders/17dinze70MWL5PmB9-Mw36zUjkrQvwb-J).
3. Extract the 7z file.

```bash
cd xrmocap_data/humman
7z x p000127_a000007.7z
```

3. Run [process_smc](./tools/process_smc.md) tool.

```bash
python tools/process_smc.py \
	--estimator_config configs/humman_mocap/mview_sperson_smpl_estimator.py \
	--smc_path xrmocap_data/humman/p000127_a000007.smc \
	--output_dir xrmocap_data/humman/p000127_a000007_output \
	--visualize
```


### Multiple People

A small test dataset for quick demo can be downloaded [here](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/example_resources/Shelf_50.zip). It contains 50 frames from the Shelf sequence, with 5 camera views calibrated and synchronized.

#### Optimization-based methods

For optimization-based approaches, it utilizes the association between 2D keypoints and generates 3D keypoints by triangulation or other methods. Taking [MVPose](../../configs/mvpose/) as an example, it can be run as

1. Download data and body model

- download data

```bash
mkdir xrmocap_data
wget https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/example_resources/Shelf_50.zip -P xrmocap_data
cd xrmocap_data/ && unzip -q Shelf_50.zip && rm Shelf_50.zip && cd ..
```
- download body model

Please refer to [Body Model Preparation](#body-model-preparation-optional)

2. Run demo

```python
python tools/mview_mperson_topdown_estimator.py \
      --estimator_config 'configs/mvpose_tracking/mview_mperson_topdown_estimator.py' \
      --image_and_camera_param 'xrmocap_data/Shelf_50/image_and_camera_param.txt' \
      --start_frame 300 \
      --end_frame 350 \
      --output_dir 'output/estimation' \
      --enable_log_file
```
If all the configuration is OK, you could see the results in `output_dir`.

#### Learning-based methods

For learning-based methods, it resorts to an end-to-end learning scheme so as to require training before inference.
Taking [MvP](../../configs/mvp/) as an example, we can download [pretrained MvP model](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_shelf-22d1b5ed_20220831.pth) and run it on Shelf_50 as:

1. Install `Deformable` package

Download the [`./ops`](https://github.com/sail-sg/mvp/tree/main/lib/models/ops) folder, rename and place the folder as `xrmocap/model/deformable`. Install `Deformable` by running:
```
cd xrmocap/model/deformable/
sh make.sh
```

2. Download data and run demo

```bash
# download data
mkdir -p xrmocap_data
wget https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/example_resources/Shelf_50.zip -P xrmocap_data
cd xrmocap_data/ && unzip -q Shelf_50.zip && rm Shelf_50.zip && cd ..

# download pretrained model
mkdir -p weight/mvp
wget https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/weight/mvp/xrmocap_mvp_shelf-22d1b5ed_20220831.pth -P weight/mvp

sh ./scripts/eval_mvp.sh 1 configs/mvp/shelf_config/mvp_shelf_50.py weight/mvp/xrmocap_mvp_shelf-22d1b5ed_20220831.pth
```

For detailed tutorials about dataset preparation, model weights and checkpoints download for learning-based methods, please refer to the [training tutorial](./tools/train_model.md) and [evaluation tutorial](./tools/eval_model.md).


## Evaluation

### Evaluate with a single GPU / multiple GPUs

#### Optimization-based methods

- Evaluate on the Shelf dataset and run the tool without tracking.

```bash
python tools/mview_mperson_evaluation.py \
      --enable_log_file \
      --evaluation_config configs/mvpose/shelf_config/eval_keypoints3d.py
```

- Evaluate on the Shelf dataset and run the tool with tracking.

```bash
python tools/mview_mperson_evaluation.py \
      --enable_log_file \
      --evaluation_config configs/mvpose_tracking/shelf_config/eval_keypoints3d.py
```

#### Learning-based methods

For learning-based methods, more details about dataset preparation, model weights and checkpoints download can be found at [evaluation tutorial](./tools/eval_model.md).

With the downloaded pretrained MvP models from [model_zoo](./benchmark.md):

```shell
sh ./scripts/val_mvp.sh ${NUM_GPUS} ${CFG_FILE} ${MODEL_PATH}
```

Example:
```shell
sh ./scripts/val_mvp.sh 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
```



### Evaluate with slurm

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `scripts/slurm_eval_mvp.sh`.

```shell
sh ./scripts/slurm_eval_mvp.sh ${PARTITION} ${NUM_GPUS} ${CFG_FILE} ${MODEL_PATH}
```

Example:
```shell
sh ./scripts/slurm_eval_mvp.sh ${PARTITION} 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
```


## Training

Training is only applicable to learning-based methods.

### Training with a single / multiple GPUs

To train the learning-based model, such as a MvP model, follow the [training tutorial](./tools/train_model.md) to prepare the datasets and pre-trained weights:

```
sh ./scripts/train_mvp.sh ${NUM_GPUS} ${CFG_FILE}
```
Example:

```
sh ./scripts/train_mvp.sh 8 configs/mvp/campus_config/mvp_campus.py

```

### Training with Slurm

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `scripts/slurm_train_mvp.sh`.

```shell
sh ./scripts/slurm_train_mvp.sh ${PARTITION} ${NUM_GPUS} ${CFG_FILE}
```
Example:
```shell
sh ./scripts/slurm_train_mvp.sh ${PARTITION} 8 configs/mvp/shelf_config/mvp_shelf.py
```


## More Tutorials

- [Introduction](./tutorials/introduction.md)
- [Config](./tutorials/config.md)
- [New dataset](./tutorials/new_dataset.md)
- [New module](./tutorials/new_module.md)
