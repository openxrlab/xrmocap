# yapf: disable
import logging
import numpy as np
from typing import Union
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.utils.triangulation_utils import (
    get_valid_views_stats, prepare_triangulate_input,
)
from .base_selector import BaseSelector

# yapf: enable


class ReprojectionErrorPointSelector(BaseSelector):

    def __init__(self,
                 target_camera_number: int,
                 triangulator: Union[BaseTriangulator, dict],
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to camera reprojection error. This selector
        will disable the worst cameras according to one reprojection result.

        Args:
            target_camera_number (int):
                For each pair of points, how many views are
                chosen.
            triangulator (Union[BaseSelector, dict]):
                Triangulator for reprojection error calculation.
                An instance or config dict.
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        if target_camera_number >= 2:
            self.target_camera_number = target_camera_number
        else:
            self.logger.error('Arg target_camera_number' +
                              ' must be no fewer than 2.\n' +
                              f'target_camera_number: {target_camera_number}')
            raise ValueError
        if isinstance(triangulator, dict):
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

    def get_selection_mask(
            self,
            points: Union[np.ndarray, list, tuple],
            init_points_mask: Union[np.ndarray, list,
                                    tuple] = None) -> np.ndarray:
        """Get a new selection mask from points and init_points_mask. This
        selector will loop triangulate points, disable the one camera with
        largest reprojection error, and loop again until there are
        self.target_camera_number left.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [n_view, n_keypoints, 2+n], n >= 0.
            init_points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [n_view, n_keypoints, 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray or a nested list of mask, in shape
                [n_view, n_keypoints, 1].
        """
        points, init_points_mask = prepare_triangulate_input(
            camera_number=len(points),
            points=points,
            points_mask=init_points_mask,
            logger=self.logger)
        n_point = points.shape[-2]
        n_view = points.shape[0]
        points2d_mask = init_points_mask.copy()
        for point_idx in range(n_point):
            selected_view = self.get_view_index(
                point=points[:, point_idx],
                init_point_mask=init_points_mask[:, point_idx])
            points2d_mask[:, point_idx] = 0
            points2d_mask[np.array(selected_view), point_idx] = 1

        points_mask_shape = points2d_mask.shape
        # log stats
        if self.verbose:
            _, stats_table = get_valid_views_stats(
                points2d_mask.reshape(n_view, -1, 1))
            self.logger.info(stats_table)
        points2d_mask = points2d_mask.reshape(*points_mask_shape)
        return points2d_mask

    def get_view_index(self, point: np.ndarray,
                       init_point_mask: np.ndarray) -> list:
        """Get a list of camera indexes. This selector will loop triangulate
        points, disable the one camera with largest reprojection error, and
        loop again until there are self.target_camera_number left.

        Args:
            point (np.ndarray):
                An ndarray of points2d, in shape
                [n_view, 2+n], n >= 0.
            init_point_mask (np.ndarray):
                An ndarray of mask, in shape [n_view, 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            list:
                A list of sorted camera indexes,
                length == self.target_camera_number.
        """
        # backup shape
        n_view = init_point_mask.shape[0]
        remain_view = np.where(init_point_mask == 1)[0]
        if n_view == 2 or len(remain_view) == self.target_camera_number:
            self.logger.warning(
                'There\'s no potential to search a sub-triangulator' +
                ' according to n_view.')
        else:
            point3d = self.triangulator.triangulate(
                points=point, points_mask=init_point_mask)
            error = self.triangulator.get_reprojection_error(
                points2d=point, points3d=point3d, points_mask=init_point_mask)
            abs_error = np.abs(error).reshape(n_view, -1)
            mean_errors = np.zeros(shape=(n_view), dtype=abs_error.dtype)
            invalid_view_count = 0
            for view_idx in range(n_view):
                if np.isnan(abs_error[view_idx]).all():
                    mean_errors[view_idx] = 1e9  # assign a large value
                    invalid_view_count += 1
                else:
                    mean_errors[view_idx] = np.nanmean(
                        abs_error[view_idx], keepdims=False)

            if self.target_camera_number + invalid_view_count > n_view:
                self.logger.error(
                    'Too many invalid views.' +
                    ' Number of valid views is lower than' +
                    f' target_camera_number {self.target_camera_number}.')
                raise ValueError
            # get mean error ignoring nan
            min_error_indexes = np.argpartition(
                mean_errors[np.where(init_point_mask == 1)[0]],
                self.target_camera_number)[:self.target_camera_number]
            remain_view = sorted(min_error_indexes.tolist())
        return remain_view
