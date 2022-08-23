import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


def interpolate_np_data(data: np.ndarray) -> np.ndarray:
    """Interpolate data in type ndarray, nan value will be set by
    interpolation.

    Args:
        data (np.ndarray):
            Points data in shape [n_frame, n_point, point_dim].

    Returns:
        np.ndarray:
            The interpolation result.
    """
    ret_data = np.apply_along_axis(__interpolate_np_nan__, 0, data)
    return ret_data


def __interpolate_np_nan__(data):
    # True if nan, False otherwise
    nan_mask = np.isnan(data)
    ret_data = np.copy(data)
    try:
        ret_data[nan_mask] = \
            np.interp(
                np.nonzero(nan_mask)[0],
                np.nonzero(~nan_mask)[0],
                data[~nan_mask])
    except ValueError:
        pass
    return ret_data


def count_masked_nan(points: np.ndarray, mask: np.ndarray) -> int:
    """Count how many points are nan after the mask applied.

    Args:
        points (np.ndarray):
            In shape [frame_n, person_n, dim].
        mask (np.ndarray):
            In shape [frame_n, person_n].

    Returns:
        int: number of np.nan whose mask is 1.
    """
    squeezed_points = np.sum(points, axis=-1, keepdims=False)
    count = np.count_nonzero(
        np.logical_and(np.isnan(squeezed_points), mask != 0))
    return count


class NanInterpolation(BaseOptimizer):

    def __init__(self,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Assign keypoints3d values by interpolation, replace nan points.

        Args:
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)

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
                'NanInterpolation only support numpy kps for now,' +
                ' the input kps has been converted to numpy.')
        total_nan_count = 0
        interp_nan_count = 0
        ret_keypoints3d = keypoints3d_np.clone()
        ret_kps_arr = ret_keypoints3d.get_keypoints()
        for person_idx in range(keypoints3d_np.get_person_number()):
            kps_arr = keypoints3d_np.get_keypoints()[:, person_idx, ...]
            mask = keypoints3d_np.get_mask()[:, person_idx, ...]
            kps_interp = interpolate_np_data(kps_arr)
            ret_kps_arr[:, person_idx, ...] = kps_interp
            # record nan
            input_nan_count = count_masked_nan(kps_arr, mask)
            output_nan_count = count_masked_nan(kps_interp, mask)
            total_nan_count += input_nan_count
            interp_nan_count += input_nan_count - output_nan_count
        ret_keypoints3d.set_keypoints(ret_kps_arr)
        if self.verbose:
            self.logger.info(
                f'\nHow many nans are found after mask: {total_nan_count}' +
                f'\nHow many nans are interpolated: {interp_nan_count}')
        return ret_keypoints3d
