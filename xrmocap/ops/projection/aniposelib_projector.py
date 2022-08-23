# yapf: disable
import logging
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import List, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.ops.projection.base_projector import BaseProjector
from xrprimer.utils.log_utils import get_logger

# yapf: enable
try:
    import aniposelib
    has_aniposelib = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_aniposelib = False
    import traceback
    import_exception = traceback.format_exc()


class AniposelibProjector(BaseProjector):
    CAMERA_CONVENTION = 'opencv'
    CAMERA_WORLD2CAM = True

    def __init__(self,
                 camera_parameters: List[FisheyeCameraParameter],
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """AniposelibProjector for points projection.

        Args:
            camera_parameters (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter.
            logger (Union[None, str, logging.Logger], optional):
                Defaults to None.
        """
        BaseProjector.__init__(self, camera_parameters)
        self.logger = get_logger(logger)
        if not has_aniposelib:
            self.logger.error(import_exception)
            raise ModuleNotFoundError(
                'Please install aniposelib to run triangulation.')

    def project(
            self,
            points: Union[np.ndarray, list, tuple],
            points_mask: Union[np.ndarray, list, tuple] = None) -> np.ndarray:
        """Project points with self.camera_parameters.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points3d, in shape
                [n_point, 3].
            points_mask (Union[np.ndarray, list, tuple], optional):
                An ndarray or a nested list of mask, in shape
                [n_point, 1].
                If points_mask[index] == 1, points[index] is valid
                for projection, else it is ignored.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray of points2d, in shape
                [n_view, n_point, 2].
        """
        points3d = np.array(points)[..., :3].reshape(-1, 3).astype(np.float64)
        n_point = points3d.shape[0]
        n_view = len(self.camera_parameters)
        points2d = np.zeros(shape=[n_view, n_point, 2], dtype=points3d.dtype)
        points_mask = np.array(points_mask).reshape(-1) \
            if points_mask is not None \
            else np.ones(shape=[n_point, ], dtype=np.uint8)
        valid_idxs = np.where(points_mask == 1)
        camera_group = self.__prepare_aniposelib_camera__()
        projected_points = camera_group.project(points3d[valid_idxs[0], :])
        points2d[:, valid_idxs[0], :] = projected_points
        return points2d

    def project_single_point(
            self, points: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Project a single point with self.camera_parameters.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a list of points3d, in shape
                [3].

        Returns:
            np.ndarray:
                An ndarray of points2d, in shape
                [n_view, 2].
        """
        points3d = np.array(points).reshape(1, 3)
        return np.squeeze(self.project(points3d), axis=1)

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
