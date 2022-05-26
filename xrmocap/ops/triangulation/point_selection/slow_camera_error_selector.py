import logging
import numpy as np
from typing import Union

from xrmocap.ops.triangulation.point_selection.camera_error_selector import \
    CameraErrorSelector  # prevent linting conflicts
from xrmocap.utils.triangulation_utils import prepare_triangulate_input
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator


class SlowCameraErrorSelector(CameraErrorSelector):

    def __init__(self,
                 target_camera_number: int,
                 triangulator: Union[BaseTriangulator, dict],
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to camera reprojection error. This selector
        will disable worst cameras one by one.

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
        super().__init__(
            target_camera_number=target_camera_number,
            triangulator=triangulator,
            verbose=verbose,
            logger=logger)

    def get_camera_indices(
            self,
            points: Union[np.ndarray, list, tuple],
            init_points_mask: Union[np.ndarray, list, tuple] = None) -> list:
        """Get a list of camera indices. This selector will loop triangulate
        points, disable the one camera with largest reprojection error, and
        loop again until there are self.target_camera_number left.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [view_number, ..., 2+n], n >= 0.
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
            list:
                A list of sorted camera indices,
                length == self.target_camera_number.
        """
        points, init_points_mask = prepare_triangulate_input(
            camera_number=len(points),
            points=points,
            points_mask=init_points_mask,
            logger=self.logger)
        # backup shape
        init_points_mask_shape = init_points_mask.shape
        view_number = init_points_mask_shape[0]
        # check if there's potential to search
        potential = True
        if view_number == 2:
            self.logger.warning(
                'There\'s no potential to search a sub-triangulator' +
                ' according to view_number.')
            potential = False
        points_mask = init_points_mask.copy()
        mean_errors = np.zeros(shape=(view_number))
        remain_cameras = [x for x in range(view_number)]
        while potential and \
                len(remain_cameras) > self.target_camera_number:
            # try to remove one camera and record mean error
            for removed_camera_index in remain_cameras:
                try_camera_list = remain_cameras.copy()
                try_camera_list.pop(remain_cameras.index(removed_camera_index))
                try_triangulator = self.triangulator[try_camera_list]
                try_points_mask = points_mask[try_camera_list, ...]
                try_points = points[try_camera_list, ...]
                try_points3d = try_triangulator.triangulate(
                    points=try_points, points_mask=try_points_mask)
                error = try_triangulator.get_reprojection_error(
                    points2d=try_points,
                    points3d=try_points3d,
                    points_mask=try_points_mask)
                abs_error = np.abs(error)
                # get mean error ignoring nan
                mean_errors[removed_camera_index] = np.nanmean(
                    abs_error.reshape(-1))
            max_indices = np.where(mean_errors == np.nanmax(mean_errors))[0]
            for camera_id in max_indices:
                remain_cameras.pop(remain_cameras.index(camera_id))
                mean_errors[camera_id] = np.nan
                if len(remain_cameras) == self.target_camera_number:
                    break
        return remain_cameras
