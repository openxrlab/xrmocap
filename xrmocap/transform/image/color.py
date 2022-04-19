from typing import Union

import numpy as np
import torch


def bgr2rgb(input_array: Union[np.ndarray, torch.Tensor], color_dim=-1):
    r_slice_list = [
        slice(None),
    ] * len(input_array.shape)
    b_slice_list = [
        slice(None),
    ] * len(input_array.shape)
    r_slice_list[color_dim] = slice(0, 1, 1)
    b_slice_list[color_dim] = slice(2, 3, 1)
    if isinstance(input_array, torch.Tensor):
        b_backup = input_array[tuple(b_slice_list)].clone()
    else:
        b_backup = input_array[tuple(b_slice_list)].copy()
    input_array[tuple(b_slice_list)] = input_array[tuple(r_slice_list)]
    input_array[tuple(r_slice_list)] = b_backup
    return input_array
