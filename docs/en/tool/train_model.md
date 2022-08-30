# Learning-based model training

## Overview

This tool takes a config file and starts trainig MvP model with train dataset.

## Preparation

1. Follow the [Prepare dataset]() tutorial to prepare the train and test data. Place the train and test data including meta data under `ROOT/xrmocap_data`.

2. (Skip if you have done this step during model evaluation) Download the [Deformable]() package, extract, rename the folder as `ROOT/xrmocap/model/deformable` and install `Deformable` by running:
```
sh deformable/make.sh
```
3. Download pre-trained backbone weights or MvP model weights from [here](). Place the model weights under `ROOT/weight`.


## Example

Start training with 8 GPUs with provided config file:

```bash
python -m torch.distributed.launch \
        --nproc_per_node= 8 \
        --use_env tool/train_model.py \
        --cfg configs/mvp/campus_config/mvp_campus.py \
```

or run the script:

```
sh scripts/train_mvp.sh
