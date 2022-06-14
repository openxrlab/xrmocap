# yapf: disable
import logging
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import Union

from xrmocap.utils.log_utils import get_logger
from xrmocap.utils.triangulation_utils import prepare_triangulate_input
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

# yapf: enable
try:
    import aniposelib
    has_aniposelib = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_aniposelib = False
    import traceback
    import_exception = traceback.format_exc()


class AniposelibTriangulator(BaseTriangulator):

    def __init__(self,
                 camera_parameters: list,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Triangulator for points triangulation, based on aniposelib.

        Args:
            camera_parameters (list):
                A list of Pinhole/FisheyeCameraParameter, or a list
                of paths to dumped Pinhole/FisheyeCameraParameters.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(camera_parameters=camera_parameters)
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
                [n_view, ..., 2+n], n >= 0.
                [...] could be [n_keypoints],
                [n_frame, n_keypoints],
                [n_frame, n_person, n_keypoints], etc.
                If length of the last dim is greater
                than 2, the redundant data will be
                dropped.
            points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [n_view, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray of points3d, in shape
                [..., 3].
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
        n_view = input_points2d_shape[0]
        points2d = points2d.reshape(n_view, -1, 2)
        points_mask = points_mask.reshape(n_view, -1, 1)
        # ignore points according to mask
        ignored_inidexes = np.where(points_mask != 1)
        points2d[ignored_inidexes[0], ignored_inidexes[1], :] = np.nan
        points3d = camera_group.triangulate(points2d)
        output_points3d_shape = np.array(input_points2d_shape[1:])
        output_points3d_shape[-1] = 3
        points3d = points3d.reshape(*output_points3d_shape)
        return points3d

    def __prepare_aniposelib_camera__(self):
        aniposelib_camera_list = []
        for cam_param in self.camera_parameters:
            if isinstance(cam_param, FisheyeCameraParameter):
                dist = [
                    cam_param.k1, cam_param.k2, cam_param.p1, cam_param.p2,
                    cam_param.k3
                ]
            else:
                dist = [
                    0.0,
                ] * 5
            args_dict = {
                'name':
                cam_param.name,
                'dist':
                dist,
                'size': [cam_param.height, cam_param.width],
                'matrix':
                cam_param.get_intrinsic(k_dim=3),
                'rvec':
                scipy_Rotation.from_matrix(cam_param.extrinsic_r).as_rotvec(),
                'tvec':
                cam_param.extrinsic_t,
            }
            camera = aniposelib.cameras.Camera(**args_dict)
            aniposelib_camera_list.append(camera)
        camera_group = aniposelib.cameras.CameraGroup(aniposelib_camera_list)
        return camera_group

    def get_reprojection_error(
            self,
            points2d: Union[np.ndarray, list, tuple],
            points3d: Union[np.ndarray, list, tuple],
            points_mask: Union[np.ndarray, list, tuple] = None) -> np.ndarray:
        """Get reprojection error between reprojected points2d and input
        points2d. Not tested yet.

        Args:
            points2d (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [n_view, ..., 2+n], n >= 0.
                [...] could be [n_keypoints],
                [n_frame, n_keypoints],
                [n_frame, n_person, n_keypoints], etc.
                Data in points2d[..., 2:] will be ignored.
            points3d (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points3d, in shape
                [..., 3+n], n >= 0.
                Data in points3d[..., 3:] will be ignored.
            points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [n_view, ..., 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray of error, in shape
                [n_view, ..., 2].
        """
        points2d, points_mask = prepare_triangulate_input(
            camera_number=len(self.camera_parameters),
            points=points2d,
            points_mask=points_mask,
            logger=self.logger)
        # backup shape for output
        input_points2d_shape = points2d.shape
        # todo: check points3d
        camera_group = self.__prepare_aniposelib_camera__()
        n_view = points_mask.shape[0]
        points2d = points2d[..., :2].copy().reshape(n_view, -1, 2)
        points3d = points3d[..., :3].copy().reshape(-1, 3)
        # ignore points according to mask
        ignored_inidexes = np.where(points_mask != 1)
        points2d[ignored_inidexes[0], ignored_inidexes[1], :] = np.nan
        points3d[ignored_inidexes[1], :] = np.nan
        errors = camera_group.reprojection_error(
            points3d, points2d, mean=False)
        output_errors_shape = np.array(input_points2d_shape)
        output_errors_shape[-1] = 2
        errors = errors.reshape(*output_errors_shape)
        return errors
