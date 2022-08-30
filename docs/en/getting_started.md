# Get started

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
├── xrmoccap
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

We assume that the cameras have been calibrated. If you want to know more about camera calibration, refer to [XRPrimer]() for more details.

### Perception Model

 -  **Prepare CamStyle models**:
You can find CamStyle model in `weight` file

### Single Person

Currently, we only provide optimization-based method for single person estimation.

```bash
xxx
```

The above code is supposed to run successfully upon you finish the installation.

### Multiple Persons

For optimization-based approaches, it does not require any pretrained model. With downloaded datasets, it can be run as

```bash
python tool/estimate_keypoints3d.py --config ./config/kps3d_estimation/shelf_config/estimate_kps3d.py
```

Some useful configs are explained here:

 - If you want to use tracing on the input sequence, you can set `use_kalman_tracking` to True in config file.

For learning-based methods, we provide model checkpoints for MvP in [model_zoo](./benchmark.md). For detailed tutorials about dataset preparation, model weights and checkpoints download for learning-based methods, please refer to the [training tutorial](./tool/train_model.md) and [evaluation tutorial](./tool/val_model.md).

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

For optimization-based methods,

```shell
# better to provide a script like beloew
# python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT} --metrics=${METRICS}
```

Evaluate on the Shelf/Campus/CMU Panoptic datasets

Example:
```shell
python xrmocap/core/evaluation/evaluate_keypoints3d.py --config ./config/kps3d_estimation/eval_kps3d_estimation.py
```

For learning-based methods, more details about dataset preparation, model weights and checkpoints download can be found at [evaluation tutorial](./tool/val_model.md).

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

### Training with a single / multiple GPUs

To train the learning-based model, such as a MvP model, follow the [training tutorial](./tool/train_model.md) to prepare the datasets and pre-trained weights:

```
sh ./scripts/train_mvp.sh ${NUM_GPUS} ${CFG_FILE}
```
Example:

```
sh ./scripts/train_mvp.sh 8 configs/mvp/campus_config/mvp_campus.py
```

### Training with Slurm

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

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
