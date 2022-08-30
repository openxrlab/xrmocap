# Learning-based model training

## Overview

This tool takes a config file and starts trainig MvP model with Shelf, Campus or CMU Panoptic dataset.

## Preparation

1. Install `Deformable` package (Skip if you have done this step during model evaluation)

Download the [`./ops`](https://github.com/sail-sg/mvp/tree/main/lib/models/ops) folder, rename and place the folder as `ROOT/xrmocap/model/deformable`. Install `Deformable` by running:
```
sh ROOT/xrmocap/model/deformable/make.sh
```

2. Prepare Dataset

Follow the [Prepare dataset](https://github.com/openxrlab/xrmocap/blob/main/docs/en/tool/prepare_dataset.md) tutorial to prepare the train and test data. Place the train and test data including meta data under `ROOT/xrmocap_data`.


3. Prepare pre-trained model weights and model checkpoints

Download pre-trained backbone weights or MvP model checkpoints from [here](https://github.com/openxrlab/xrmocap/blob/mvp_doc_dev/configs/mvp/README.md). Place the model weights under `ROOT/weight`.

4. Prepare config files

Modify the config files in `ROOT/configs/mvp` if needed. Make sure the directories in config files match the directories and file names for your dataset and pre-traind weights.

## Example

Start training with 8 GPUs with provided config file for Campus dataset:

```bash
python -m torch.distributed.launch \
        --nproc_per_node= 8 \
        --use_env tool/train_model.py \
        --cfg configs/mvp/campus_config/mvp_campus.py \
```

or directly run the script:

```
sh ROOT/scripts/train_mvp.sh
