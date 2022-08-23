import logging
import numpy as np
from scipy import signal
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


def median_filter_data(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Perform a median filter on an ndarray, along the first axis of data.

    Args:
        data (np.ndarray):
            Points data in shape [n_frame, n_point, point_dim].
        kernel_size (int):
            Size of the filter kernel.

    Returns:
        np.ndarray:
            The filterd result.
    """
    ret_data = np.apply_along_axis(
        __median_filter__, 0, data, kernel_size=kernel_size)
    return ret_data


def __median_filter__(data, kernel_size):
    pad_size = kernel_size + int((kernel_size + 1) / 2)
    padded_data = np.pad(data, (pad_size, pad_size), mode='reflect')
    padded_filterd_data = signal.medfilt(padded_data, kernel_size=kernel_size)
    return padded_filterd_data[pad_size:-pad_size]


class MedianSmooth(BaseOptimizer):

    def __init__(self,
                 kernel_size: int = 7,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Perform a median filter on a Keypoints instance.

        Args:
            kernel_size (int, optional):
                Size of the filter kernel. Defaults to 7.
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        if kernel_size % 2 == 0:
            self.logger.error('kernel_size of MedianFilter should be odd.')
            raise ValueError
        self.kernel_size = kernel_size

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **kwargs: dict) -> Keypoints:
        """Forward function of keypoints3d optimizer.

        Args:
            keypoints3d (Keypoints): Input keypoints3d.
        kwargs:
            Redundant keyword arguments to be
            ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        if keypoints3d.dtype == 'numpy':
            keypoints3d_np = keypoints3d
        else:
            keypoints3d_np = keypoints3d.to_numpy()
            self.logger.warning(
                'MedianFilter only support numpy kps for now,' +
                ' the input kps has been converted to numpy.')
        ret_keypoints3d = keypoints3d_np.clone()
        ret_kps_arr = ret_keypoints3d.get_keypoints()
        for person_idx in range(keypoints3d_np.get_person_number()):
            kps_arr = keypoints3d_np.get_keypoints()[:, person_idx, ...]
            kps_interp = median_filter_data(
                kps_arr, kernel_size=self.kernel_size)
            ret_kps_arr[:, person_idx, ...] = kps_interp
        ret_keypoints3d.set_keypoints(ret_kps_arr)
        return ret_keypoints3d
