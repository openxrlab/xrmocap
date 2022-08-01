# yapf: disable
import logging
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import List, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator
from xrprimer.utils.log_utils import get_logger

from xrmocap.utils.triangulation_utils import prepare_triangulate_input
from ..projection.aniposelib_projector import AniposelibProjector

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# yapf: enable
try:
    import aniposelib
    has_aniposelib = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_aniposelib = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class AniposelibTriangulator(BaseTriangulator):
    CAMERA_CONVENTION = 'opencv'
    CAMERA_WORLD2CAM = True

    def __init__(self,
                 camera_parameters: List[FisheyeCameraParameter],
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Triangulator for points triangulation, based on aniposelib.

        Args:
            camera_parameters (List[FisheyeCameraParameter]):
                A list of Pinhole/FisheyeCameraParameter, or a list
                of paths to dumped Pinhole/FisheyeCameraParameters.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        super().__init__(camera_parameters=camera_parameters, logger=logger)
        if not has_aniposelib:
            self.logger.error(import_exception)
            raise ModuleNotFoundError(
                'Please install aniposelib to run triangulation.')

    def set_cameras(
        self, camera_parameters: List[Union[PinholeCameraParameter,
                                            FisheyeCameraParameter]]
    ) -> None:
        """Set cameras for this triangulator.

        Args:
            camera_parameters (List[Union[PinholeCameraParameter,
                                          FisheyeCameraParameter]]):
                A list of PinholeCameraParameter, or a list
                of FisheyeCameraParameter.
        """
        if len(camera_parameters) > 0 and \
                isinstance(camera_parameters[0], str):
            self.logger.error('camera_parameters must be a list' +
                              ' of camera parameter instances, not strs.')
            raise TypeError
        super().set_cameras(camera_parameters=camera_parameters)

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
        ignored_indexes = np.where(points_mask != 1)
        points2d[ignored_indexes[0], ignored_indexes[1], :] = np.nan
        points3d = camera_group.triangulate(points2d)
        output_points3d_shape = np.array(input_points2d_shape[1:])
        output_points3d_shape[-1] = 3
        points3d = points3d.reshape(*output_points3d_shape)
        return points3d

    def __prepare_aniposelib_camera__(self) -> aniposelib.cameras.CameraGroup:
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
            args_dict = dict(
                name=cam_param.name,
                dist=dist,
                size=[cam_param.height, cam_param.width],
                matrix=cam_param.get_intrinsic(k_dim=3),
                rvec=scipy_Rotation.from_matrix(
                    cam_param.extrinsic_r).as_rotvec(),
                tvec=cam_param.extrinsic_t,
            )
            camera = aniposelib.cameras.Camera(**args_dict)
            aniposelib_camera_list.append(camera)
        camera_group = aniposelib.cameras.CameraGroup(aniposelib_camera_list)
        return camera_group

    def get_reprojection_error(
        self,
        points2d: Union[np.ndarray, list, tuple],
        points3d: Union[np.ndarray, list, tuple],
        points_mask: Union[np.ndarray, list, tuple] = None,
        reduction: Literal['mean', 'sum', 'none'] = 'none'
    ) -> Union[np.ndarray, float]:
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
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the error to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'none'.

        Returns:
            Union[np.ndarray, float]:
                If reduction is None, an ndarray of error, in shape
                [n_view, ..., 2].
                If reduction is sum or mean, a float scalar is returned.
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
        points_mask = points_mask.copy().reshape(n_view, -1, 1)
        # ignore points according to mask
        ignored_indexes = np.where(points_mask != 1)
        points2d[ignored_indexes[0], ignored_indexes[1], :] = np.nan
        errors = camera_group.reprojection_error(
            points3d, points2d, mean=False)

        if reduction == 'none':
            output_errors_shape = np.array(input_points2d_shape)
            output_errors_shape[-1] = 2
            errors = errors.reshape(*output_errors_shape)
            return errors
        elif reduction == 'mean':
            axis = [x for x in range(len(errors.shape))]
            errors = np.mean(errors, axis=tuple(axis))
        else:  # sum
            axis = [x for x in range(len(errors.shape))]
            errors = np.sum(errors, axis=tuple(axis))
        return errors

    def get_projector(self) -> AniposelibProjector:
        """Get an AniposelibProjector according to parameters of this
        triangulator.

        Returns:
            AniposelibProjector
        """
        projector = AniposelibProjector(
            camera_parameters=self.camera_parameters, logger=self.logger)
        return projector
