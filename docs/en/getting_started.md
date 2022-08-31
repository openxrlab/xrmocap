# Getting started

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

Download the above resources and arrange them in the following file structure:

```text
xrmocap
├── xrmocap
├── docs
├── tests
├── tools
├── configs
└── data
    └── body_models
        ├── smpl_mean_params.npz
        └── smpl
            ├── SMPL_FEMALE.pkl
            ├── SMPL_MALE.pkl
            └── SMPL_NEUTRAL.pkl
```

## Inference / Demo

We provide a demo script to estimate SMPL parameters for single-person or multi-person from multi-view synchronized input images or videos. With this demo script, you only need to choose a method, we currently support two types of methods, namely, optimization-based approaches and end-to-end learning algorithms, specify a few arguments, and then you can get the estimated results.

We assume that the cameras have been calibrated. If you want to know more about camera calibration, refer to [XRPrimer](https://github.com/openxrlab/xrprimer/blob/main/docs/en/tool/calibrate_pinhole_cameras.md) for more details.


### Perception Model

 -  **Prepare CamStyle models**:

```
sh scripts/download_weight.sh
```
You could find `resnet50_reid_camstyle.pth.tar` in `weight` file.

### Single Person

Currently, we only provide optimization-based method for single person estimation.

1. Download an SMC file from [humman dataset](https://drive.google.com/drive/folders/17dinze70MWL5PmB9-Mw36zUjkrQvwb-J).
2. Extract the 7z file.

```bash
7z x p000127_a000007.7z
```

3. Run [process_smc](./tools/process_smc.md) tool.

```bash
mkdir xrmocap_data/humman
python tools/process_smc.py \
	--estimator_cofig configs/humman_mocap/mview_sperson_smpl_estimator.py \
	--smc_path p000127_a000007.smc \
	--output_dir xrmocap_data/humman/p000127_a000007_output \
	--visualize
```

The above code is supposed to run successfully upon you finish the installation.


### Multiple People

A small test dataset for quick inference and demo can be downloaded from [here](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrmocap/example_resources/Shelf_50.zip). It contains 50 frames from the Shelf sequence, with 5 camera views calibrated and synchronized.

#### Optimization-based methods

For optimization-based approaches, it does not require any pretrained model. Taking [MVPose](https://zju3dv.github.io/mvpose/) as an example, it can be run as

```bash
Coming soon!
```

Some useful configs are explained here:

 - If you want to use tracing on the input sequence, you can set `use_kalman_tracking` to True in config file.

#### Learning-based methods

For learning-based methods, we provide model checkpoints for MvP in [model_zoo](./benchmark.md). For detailed tutorials about dataset preparation, model weights and checkpoints download for learning-based methods, please refer to the [training tutorial](./tools/train_model.md) and [evaluation tutorial](./tools/eval_model.md).

With the downloaded pretrained MvP models:

```shell
sh ./scripts/val_mvp.sh ${NUM_GPUS} ${CFG_FILE} ${MODEL_PATH}
```

Example:
```shell
sh ./scripts/val_mvp.sh 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
```


## Evaluation

We provide pretrained models in the respective method folders in [config](config).

### Evaluate with a single GPU / multiple GPUs

#### Optimization-based methods

Evaluate on the Shelf/Campus/CMU Panoptic datasets

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
sh ./scripts/slurm_eval_mvp.sh MyPartition 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
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
sh ./scripts/slurm_train_mvp.sh MyPartition 8 configs/mvp/shelf_config/mvp_shelf.py
```


## More Tutorials

- [Introduction](./tutorials/introduction.md)
- [Config](./tutorials/config.md)
- [New dataset](./tutorials/new_dataset.md)
