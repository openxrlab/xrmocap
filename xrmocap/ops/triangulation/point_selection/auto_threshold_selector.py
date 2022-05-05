import logging
from typing import Union

import numpy as np

from xrmocap.utils.triangulation_utils import (
    get_valid_views_stats,
    prepare_triangulate_input,
)
from .base_selector import BaseSelector


class AutoThresholdSelector(BaseSelector):

    def __init__(self,
                 start: float = 0.95,
                 stride: float = -0.05,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to confidence, while the threshold of
        confidence is selected automatically according to input data. Try the
        largest keypoints_threshold by loop, which can be represented as
        start+n*stride and makes number of valid views >= 2. If you need a
        lower bound of keypoints_threshold, pass the output of
        ManualThresholdSelector as init_points_mask.

        Args:
            start (float, optional):
                Init threshold, should be in (0, 1].
                Defaults to 0.95.
            stride (float, optional):
                Step of one loop, should be in (-start, 0).
                Defaults to -0.05.
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        if start > 0 and start <= 1:
            self.start = start
        else:
            self.logger.error('Arg start must be in range (0, 1].\n' +
                              f'start: {start}')
            raise ValueError
        if stride > (0 - start) and stride < 0:
            self.stride = stride
        else:
            self.logger.error('Arg stride must be in range (-start, 0).\n' +
                              f'stride: {stride}')
            raise ValueError

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
                [...] could be [keypoint_num],
                [frame_num, keypoint_num],
                [frame_num, person_num, keypoint_num], etc.
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
        points2d_conf = points2d_conf.reshape(view_number, -1)
        init_points_mask = init_points_mask.reshape(view_number, -1)
        # check if there's potential to search
        view_count = np.sum(init_points_mask, axis=0)
        # np.nan will be ignored, only figure matters
        potential = True
        if np.any(view_count < 2):
            self.logger.warning(
                'There\'s no potential to search a higher threshold' +
                ' according to init_points_mask.')
            potential = False
        threshold = self.start
        while potential and threshold >= 0:
            pair_fail = False
            filtered_mask = (points2d_conf >= threshold).astype(
                np.uint8) * init_points_mask
            view_count = np.sum(filtered_mask, axis=0)
            if np.any(view_count < 2):
                pair_fail = True
            if pair_fail:
                threshold += self.stride
            else:
                break
        points2d_mask = (points2d_conf >= threshold).astype(
            np.uint8) * init_points_mask
        points2d_mask = points2d_mask[:, :, np.newaxis]
        # log stats
        if self.verbose:
            self.logger.info(f'Auto points threshold found: {threshold}')
            _, stats_table = get_valid_views_stats(points2d_mask)
            self.logger.info(stats_table)
        points2d_mask = points2d_mask.reshape(*init_points_mask_shape)
        return points2d_mask
