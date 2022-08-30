# Learning-based model validation

## Overview

This tool takes a config file and pre-trained MvP weights and performs evaluation on Shelf, Campus or CMU Panoptic dataset.

## Preparation
1. Follow the [Prepare dataset]() tutorial to prepare the train and test data. Place the train and test data including meta data under `ROOT/xrmocap_data`.

2. (Skip if you have done this step during model training) Download the [Deformable]() package, extract, rename the folder as `ROOT/xrmocap/model/deformable` and install `Deformable` by running:
```
sh deformable/make.sh
```
3. Download pre-trained backbone weights or MvP model weights from [here](). Place the model weights under `ROOT/weight`.


### Example

Start evaluation with 8 GPUs with provided config file and pre-trained weights:

```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env tool/val_model.py \
    --cfg configs/mvp/shelf_config/mvp_shelf.py \
    --model_path weight/xrmocap_mvp_shelf.pth.tar
```

or run the script:

```
sh scripts/val_mvp.sh
```
