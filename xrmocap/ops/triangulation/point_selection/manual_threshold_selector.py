# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.utils.triangulation_utils import (
    get_valid_views_stats, prepare_triangulate_input,
)
from .base_selector import BaseSelector

# yapf: enable


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
                [n_view, ..., 3]. Confidence of points is in
                [n_view, ..., 2:3].
                [...] could be [n_keypoints],
                [n_frame, n_keypoints],
                [n_frame, n_person, n_keypoints], etc.
            init_points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [n_view, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray or a nested list of mask, in shape
                [n_view, ..., 1].
        """
        points, init_points_mask = prepare_triangulate_input(
            camera_number=len(points),
            points=points,
            points_mask=init_points_mask,
            logger=self.logger)
        # backup shape
        init_points_mask_shape = init_points_mask.shape
        n_view = init_points_mask_shape[0]
        # points with confidence
        points2d_conf = points[..., 2:3].copy()
        points2d_conf = points2d_conf.reshape(n_view, -1, 1)
        points2d_mask = init_points_mask.reshape(n_view, -1, 1).copy()
        # ignore points according to threshold
        points2d_mask = (points2d_conf >= self.threshold).astype(
            np.uint8) * points2d_mask
        # log stats
        if self.verbose:
            _, stats_table = get_valid_views_stats(points2d_mask)
            self.logger.info(stats_table)
        points2d_mask = points2d_mask.reshape(*init_points_mask_shape)
        return points2d_mask
