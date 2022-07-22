import logging
import numpy as np
import torch
from typing import Union

from .base_image_transform import BaseImageTransform


class BGR2RGB(BaseImageTransform):

    def __init__(self,
                 color_dim: int = -1,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Convert image array of any shape between BGR and RGB.

        Args:
            color_dim (int, optional):
                Which dim is the color channel. Defaults to -1.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseImageTransform.__init__(self, logger=logger)
        self.color_dim = color_dim

    def forward(
        self, input: Union[np.ndarray,
                           torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Forward function of BGR2RGB.

        Args:
            input (Union[np.ndarray, torch.Tensor]):
                An array of images. The shape could be:
                [h, w, n_ch], [n_frame, h, w, n_ch],
                [n_view, n_frame, h, w, n_ch], etc.

        Returns:
            Union[np.ndarray, torch.Tensor]
        """
        return bgr2rgb(input, self.color_dim)


def bgr2rgb(input_array: Union[np.ndarray, torch.Tensor],
            color_dim: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """Convert image array of any shape between BGR and RGB.

    Args:
        input_array (Union[np.ndarray, torch.Tensor]):
            An array of images. The shape could be:
            [h, w, n_ch], [n_frame, h, w, n_ch],
            [n_view, n_frame, h, w, n_ch], etc.
        color_dim (int, optional):
            Which dim is the color channel. Defaults to -1.

    Returns:
        Union[np.ndarray, torch.Tensor]:
            Same type as the input.
    """
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
