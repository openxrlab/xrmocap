# Learning-based model training

- [Overview](#overview)
- [Preparation](#preparation)
- [Example](#example)

## Overview

This tool takes a config file and starts trainig MvP model with Shelf, Campus or CMU Panoptic dataset.

## Preparation

1. Download and install the `Deformable` package (Skip if you have done this step during model evaluation)

Run the script:
```
sh scripts/download_install_deformable.sh
```

2. Prepare Datasets

Follow the [dataset tool](./prepare_dataset.md) tutorial to prepare the train and test data. Some pre-processed datasets are available for download [here](../dataset_preparation.md). Place the `trainset_pesudo_gt` and `testset` file including meta data under `ROOT/xrmocap_data/DATASET`.


3. Prepare pre-trained model weights and model checkpoints

Download pre-trained backbone weights or MvP model checkpoints from [here](../../../configs/mvp/README.md). Place the model weights under `ROOT/weight/mvp`.

4. Prepare config files

Modify the config files in `ROOT/configs/mvp` if needed. Make sure the directories in config files match the directories and file names for your datasets and pre-traind weights.

The final file structure ready for training would be like:

```text
xrmocap
├── xrmocap
├── tools
├── configs
├── weight
|   └── mvp
|       ├── xrmocap_mvp_campus-[version].pth
|       ├── xrmocap_mvp_shelf-[version].pth
|       ├── xrmocap_mvp_panoptic_5view-[version].pth
|       ├── xrmocap_mvp_panoptic_3view_3_12_23-[version].pth
|       └── xrmocap_pose_resnet50_panoptic-[version].pth
└── xrmocap_data
    ├── Shelf
    |   ├── xrmocap_meta_testset
    |   ├── xrmocap_meta_trainset_pesudo_gt
    |   ├── Camera0
    |   ├── ...
    |   └── Camera4
    ├── CampusSeq1
    └── panoptic
        ├── 160906_band4
        ├── 160906_ian5
        ├── ...
        └── 160906_pizza1

```

## Example

Start training with 8 GPUs with provided config file for Campus dataset:

```bash
python -m torch.distributed.launch \
        --nproc_per_node= 8 \
        --use_env tools/train_model.py \
        --cfg configs/mvp/campus_config/mvp_campus.py \
```

Alternatively, you can also run the script directly:

```
sh ROOT/scripts/train_mvp.sh ${NUM_GPUS} ${CFG_FILE}
```
Example:

```
sh ROOT/scripts/train_mvp.sh 8 configs/mvp/campus_config/mvp_campus.py
```
If you encounter a RuntimeError saying that dataloader's workers are out of shared memory, try changing the `workers` to 1 in the config file.

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script:
```shell
sh ROOT/scripts/slurm_train_mvp.sh ${PARTITION} ${NUM_GPUS} ${CFG_FILE}
```
Example:
```shell
sh ROOT/scripts/slurm_train_mvp.sh MyPartition 8 configs/mvp/shelf_config/mvp_shelf.py
```
