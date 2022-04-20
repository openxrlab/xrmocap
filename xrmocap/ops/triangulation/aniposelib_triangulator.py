import logging
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

from xrmocap.ops.triangulation.builder import TRIANGULATORS
from xrmocap.utils.log_utils import get_logger
from xrmocap.utils.triangulation_utils import prepare_triangulate_input

try:
    import aniposelib
    has_aniposelib = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_aniposelib = False
    import traceback
    import_exception = traceback.format_exc()


@TRIANGULATORS.register_module(name=('AniposelibTriangulator'))
class AniposelibTriangulator(BaseTriangulator):

    def __init__(self,
                 camera_parameters: list,
                 camera_convention: str = 'opencv',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Triangulator for points triangulation, based on aniposelib.

        Args:
            camera_parameters (list):
                A list of PinholeCameraParameter, or a list
                of paths to dumped PinholeCameraParameters.
            camera_convention (str, optional):
                Expected convention name of cameras.
                If camera_parameters do not match expectation,
                convert them to the correct convention.
                Defaults to 'opencv'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(
            camera_parameters=camera_parameters,
            camera_convention=camera_convention)
        self.logger = get_logger(logger)
        if not has_aniposelib:
            self.logger.error(import_exception)
            raise ModuleNotFoundError(
                'Please install aniposelib to run triangulation.')

    def triangulate(
            self,
            points: Union[np.ndarray, list, tuple],
            points_mask: Union[np.ndarray, list, tuple] = None) -> np.ndarray:
        """Triangulate points with self.camera_parameters.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [view_number, ..., 2+n], n >= 0.
                If length of the last dim is greater
                than 2, the redundant data will be
                dropped.
            points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [view_number, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray of points3d, in shape
                [view_number, ..., 3].
        """
        points, points_mask = prepare_triangulate_input(
            camera_number=len(self.camera_parameters),
            points=points,
            points_mask=points_mask,
            logger=self.logger)
        camera_group = self.__prepare_aniposelib_camera__()
        # split points data and redundant data
        points2d = points[..., :2].copy()
        # backup shape for output
        input_points2d_shape = points2d.shape
        view_number = input_points2d_shape[0]
        points2d = points2d.reshape(view_number, -1, 2)
        points_mask = points_mask.reshape(view_number, -1, 1)
        # ignore points according to mask
        ignored_indices = np.where(points_mask != 1)
        points2d[ignored_indices[0], ignored_indices[1], :] = np.nan
        points3d = camera_group.triangulate(points2d)
        output_points3d_shape = np.array(input_points2d_shape[1:])
        output_points3d_shape[-1] = 3
        points3d = points3d.reshape(*output_points3d_shape)
        return points3d

    def __prepare_aniposelib_camera__(self):
        aniposelib_camera_list = []
        for cam_param in self.camera_parameters:
            param_dict = cam_param.to_dict()
            args_dict = {
                'name':
                param_dict['name'],
                'dist': [
                    param_dict['k1'], param_dict['k2'], param_dict['p1'],
                    param_dict['p2'], param_dict['k3']
                ],
                'size': [param_dict['height'], param_dict['width']],
                'matrix':
                cam_param.get_intrinsic(k_dim=3),
                'rvec':
                scipy_Rotation.from_matrix(
                    param_dict['extrinsic_r']).as_rotvec(),
                'tvec':
                param_dict['extrinsic_t'],
            }
            camera = aniposelib.cameras.Camera(**args_dict)
            aniposelib_camera_list.append(camera)
        camera_group = aniposelib.cameras.CameraGroup(aniposelib_camera_list)
        return camera_group

    def get_reprojection_error(
            self,
            points2d: Union[np.ndarray, list, tuple],
            points3d: Union[np.ndarray, list, tuple],
            points_mask: Union[np.ndarray, list, tuple] = None) -> float:
        """Get reprojection error between reprojected points2d and input
        points2d. Not tested yet.

        Args:
            points2d (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [view_number, ..., 2+n], n >= 0.
                Data in points2d[..., 2:] will be ignored.
            points3d (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points3d, in shape
                [..., 3+n], n >= 0.
                Data in points3d[..., 3:] will be ignored.
            points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [view_number, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            float: The error value.
        """
        points2d, points_mask = prepare_triangulate_input(
            points2d, points_mask)
        # todo: check points3d
        camera_group = self.__prepare_aniposelib_camera__()
        view_number = points_mask.shape[0]
        points2d = points2d[..., :2].reshape(view_number, -1, 2)
        points3d = points3d[..., :3].reshape(-1, 3)
        # ignore points according to mask
        ignored_indices = np.where(points_mask != 1)
        points2d[ignored_indices[0], ignored_indices[1], :] = np.nan
        points3d[ignored_indices[1], :] = np.nan
        errors = camera_group.reprojection_error(points3d, points2d)
        return errors
