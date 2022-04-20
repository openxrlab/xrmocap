import logging
from typing import Union

import numpy as np

from xrmocap.ops.triangulation.builder import POINTSELECTORS
from .base_selector import BaseSelector


@POINTSELECTORS.register_module(name=('AutoThresholdSelector'))
class AutoThresholdSelector(BaseSelector):

    def __init__(self,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to confidence, while the threshold of
        confidence is selected automatically according to input data.

        Args:
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)

    def get_selection_mask(
            self,
            points: Union[np.ndarray, list, tuple],
            init_points_mask: Union[np.ndarray, list,
                                    tuple] = None) -> np.ndarray:
        """Get a new selection mask from points and init_points_mask. This
        selector will assure that for each valid data pair, at least two views
        remain.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [view_number, ..., 3]. Confidence of points is in
                [view_number, ..., 2:3].
            init_points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [view_number, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray or a nested list of mask, in shape
                [view_number, ..., 1].
        """
        raise NotImplementedError
