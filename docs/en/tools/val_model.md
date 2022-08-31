# Learning-based model evaluation

- [Overview](#overview)
- [Preparation](#preparation)
- [Example](#example)

## Overview

This tool takes a config file and MvP model checkpoints and performs evaluation on Shelf, Campus or CMU Panoptic dataset.

## Preparation

1. Install `Deformable` package (Skip if you have done this step during model training)

Download the [`./ops`](https://github.com/sail-sg/mvp/tree/main/lib/models/ops) folder, rename and place the folder as `ROOT/xrmocap/model/deformable`. Install `Deformable` by running:
```
sh ROOT/xrmocap/model/deformable/make.sh
```

2. Prepare Datasets

Follow the [dataset tool](./prepare_dataset.md) tutorial to prepare the train and test data. Some pre-processed datasets are available for download [here](../dataset_preparation.md). Place the `trainset_pesudo_gt` and `testset` data including meta data under `ROOT/xrmocap_data`.


3. Prepare pre-trained model weights and model checkpoints

Download pre-trained backbone weights or MvP model checkpoints from [here](../../../configs/mvp/README.md). Place the model weights under `ROOT/weight`.

4. Prepare config files

Modify the config files in `ROOT/configs/mvp` if needed. Make sure the directories in config files match the directories and file names for your datasets and pre-traind model weights.

The final file structure ready for evaluation would be like:

```text
xrmocap
├── xrmoccap
├── tools
├── configs
└── weight
    ├── xrmocap_mvp_campus.pth.tar
    ├── xrmocap_mvp_shelf.pth.tar
    ├── xrmocap_mvp_panoptic_5view.pth.tar
    ├── xrmocap_mvp_panoptic_3view_3_12_23.pth.tar
    └── xrmocap_pose_resnet50_panoptic.pth.tar
└── xrmocap_data
    └── meta  
        └── shelf
            └── xrmocap_meta_testset
        ├── campus
        └── panoptic
    ├── Shelf
    ├── CampusSeq1
    └── panoptic
        ├── 160906_band4
        ├── 160906_ian5
        ├── ...
        └── 160906_pizza1

```

### Example

Start evaluation with 8 GPUs with provided config file and pre-trained weights for Shelf dataset:

```shell
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env tools/val_model.py \
    --cfg configs/mvp/shelf_config/mvp_shelf.py \
    --model_path weight/xrmocap_mvp_shelf.pth.tar
```

Alternatively, you can also run the script directly:

```shell
sh ROOT/scripts/val_mvp.sh ${NUM_GPUS} ${CFG_FILE} ${MODEL_PATH}
```

Example:
```shell
sh ROOT/scripts/val_mvp.sh 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
```

If you can run XRMoCap on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script:
```shell
sh ROOT/scripts/slurm_eval_mvp.sh ${PARTITION} ${NUM_GPUS} ${CFG_FILE} ${MODEL_PATH}
```
Example:
```shell
sh ROOT/scripts/slurm_eval_mvp.sh MyPartition 8 configs/mvp/shelf_config/mvp_shelf.py weight/xrmocap_mvp_shelf.pth.tar
```
