# Learning-based model training

## Overview



## Arguments

## Example 

Start training with 8 GPUs with provided config files:

```bash 
python -m torch.distributed.launch \
        --nproc_per_node= 8 \
        --use_env tool/train_model.py \
        --cfg configs/mvp/campus_config/mvp_campus.py \
```
