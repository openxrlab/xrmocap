import logging
import numpy as np
import torch
from typing import Union
from xrprimer.utils.log_utils import get_logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def convert_bbox(
    data: Union[torch.Tensor, np.ndarray],
    src: Literal['xyxy', 'xywh'],
    dst: Literal['xyxy', 'xywh'],
    logger: Union[None, str, logging.Logger] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Convert bbox2d following the mapping correspondence between src and dst
    keypoints definition. Supported conventions by now: xyxy, xywh.

    Args:
        data (Union[torch.Tensor, np.ndarray]):
            Bbox2d data. The last dim should be (x, y, x, y, score),
            or (x, y, w, h, score), and score is optional.
        src (str):
            Bbox type, xyxy or xywh.
        dst (str):
            Bbox type, xyxy or xywh.
        logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

    Raises:
        ValueError: Either src or dst is not in defined conventions.

    Returns:
        Union[torch.Tensor, np.ndarray]:
            The converted bbox2d in the same shape and type as input data.
    """
    logger = get_logger(logger)
    if src == dst:
        logger.warning('Src convention equals to dst convention.')
        return data
    if isinstance(data, torch.Tensor):

        def abs_func(data):
            return torch.abs(input=data)

        def min_func(data):
            return torch.min(input=data, dim=-1, keepdim=False)[0]

        def clone_func(data):
            return torch.clone(data)
    elif isinstance(data, np.ndarray):

        def abs_func(data):
            return np.absolute(data)

        def min_func(data):
            return np.min(data, axis=-1, keepdims=False)

        def clone_func(data):
            return np.copy(data)

    if src == 'xyxy':
        w = abs_func(data[..., 0] - data[..., 2])
        h = abs_func(data[..., 1] - data[..., 3])
        x_min = min_func(data[..., slice(0, 4, 2)])
        y_min = min_func(data[..., slice(1, 4, 2)])
    elif src == 'xywh':
        w = data[..., 2]
        h = data[..., 3]
        x_min = data[..., 0]
        y_min = data[..., 1]
    else:
        raise ValueError('Wrong source name.')
    output = clone_func(data)
    output[..., 0] = x_min
    output[..., 1] = y_min
    if dst == 'xyxy':
        output[..., 2] = x_min + w
        output[..., 3] = y_min + h
    elif dst == 'xywh':
        output[..., 2] = w
        output[..., 3] = h
    else:
        raise ValueError('Wrong destination name.')
    return output
