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
You can find `resnet50_reid_camstyle.pth.tar` in `weight` file.

### Single Person

Currently, we only provide optimization-based method for single person estimation.

```bash
# @gy
```

The above code is supposed to run successfully upon you finish the installation.

### Multiple People

#### Optimization-based methods

For optimization-based approaches, it does not require any pretrained model. Taking [MVPose](https://zju3dv.github.io/mvpose/) as an example, it can be run as

```bash
Coming soon!
```

Some useful configs are explained here:

 - If you want to use tracing on the input sequence, you can set `use_kalman_tracking` to True in config file.

#### Learning-based methods

For learning-based methods, we provide pretrained models in [model_zoo](), it can be downloaded and run the script as below.

```bash
sh script/eval_mvp.sh
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

For learning-based methods, with the downloaded pretrained models from [model_zoo]():

```bash
sh script/slurm_eval_mvp.sh
```

### Evaluate with slurm

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT} --metrics ${METRICS}
```

Example:
```shell
./tools/slurm_test.sh my_partition test_hmr configs/hmr/resnet50_hmr_pw3d.py work_dirs/hmr work_dirs/hmr/latest.pth 8 --metrics pa-mpjpe mpjpe
```


## Training

Training is only applicable to learning-based methods.

### Training with a single / multiple GPUs

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```

Example: using 1 GPU to train MvP.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### Training with Slurm

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```


## More Tutorials

- [Introduction](./tutorials/introduction.md)
- [Config](./tutorials/config.md)
- [New dataset](./tutorials/new_dataset.md)
