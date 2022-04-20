import logging
from typing import Union

import numpy as np
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

from xrmocap.ops.triangulation.builder import POINTSELECTORS
from xrmocap.ops.triangulation.point_selection.base_selector import \
    BaseSelector  # not in registry, cannot be built


@POINTSELECTORS.register_module(name=('CameraErrorSelector'))
class CameraErrorSelector(BaseSelector):

    def __init__(self,
                 target_camera_number: int,
                 triangulator: Union[BaseTriangulator, dict],
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to camera reprojection error.

        Args:
            target_camera_number (int):
                For each pair of points, how many views are
                chosen.
                Defaults to True.
            triangulator (Union[BaseSelector, dict]):
                Triangulator for reprojection error calculation.
                An instance or config dict.
                Defaults to True.
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        raise NotImplementedError

    def get_selection_mask(
            self,
            points: Union[np.ndarray, list, tuple],
            init_points_mask: Union[np.ndarray, list,
                                    tuple] = None) -> np.ndarray:
        """Get a new selection mask from points and init_points_mask. This
        selector will disable some cameras for every point pair, only use
        self.target_camera_number cameras to triangulate.

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
