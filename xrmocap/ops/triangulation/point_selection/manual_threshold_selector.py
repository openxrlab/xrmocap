import logging
from typing import Union

import numpy as np

from xrmocap.ops.triangulation.builder import POINTSELECTORS
from xrmocap.ops.triangulation.point_selection.base_selector import \
    BaseSelector  # not in registry, cannot be built
from xrmocap.utils.triangulation_utils import (
    get_valid_views_stats,
    prepare_triangulate_input,
)


@POINTSELECTORS.register_module(name=('ManualThresholdSelector'))
class ManualThresholdSelector(BaseSelector):

    def __init__(self,
                 threshold: float = 0.0,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to confidence. If confidence of a point >=
        threshold, it will be selected.

        Args:
            threshold (float, optional):
                Threshold of point selection.
                Defaults to 0.0.
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        self.threshold = threshold

    def get_selection_mask(
            self,
            points: Union[np.ndarray, list, tuple],
            init_points_mask: Union[np.ndarray, list,
                                    tuple] = None) -> np.ndarray:
        """Get a new selection mask from points and init_points_mask.

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
        points, init_points_mask = prepare_triangulate_input(
            camera_number=len(points),
            points=points,
            points_mask=init_points_mask,
            logger=self.logger)
        # backup shape
        init_points_mask_shape = init_points_mask.shape
        view_number = init_points_mask_shape[0]
        # points with confidence
        points2d_conf = points[..., 2:3].copy()
        points2d_conf = points2d_conf.reshape(view_number, -1, 1)
        points2d_mask = init_points_mask.reshape(view_number, -1, 1).copy()
        # ignore points according to threshold
        ignored_indices = np.where(points2d_conf < self.threshold)
        points2d_mask[ignored_indices[0], ignored_indices[1], :] = 0
        points2d_mask = points2d_mask.reshape(*init_points_mask_shape)
        # log stats
        if self.verbose:
            _, stats_table = get_valid_views_stats(points2d_mask)
            self.logger.info(stats_table)
        return points2d_mask
