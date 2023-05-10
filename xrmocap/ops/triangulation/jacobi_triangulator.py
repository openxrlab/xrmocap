# yapf: disable
import numpy as np
from typing import List, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

from xrmocap.utils.triangulation_utils import prepare_triangulate_input

# yapf: enable


class JacobiTriangulator(BaseTriangulator):

    def __init__(self,
                 camera_parameters: List[FisheyeCameraParameter] = [],
                 maxIter_time=20,
                 update_tolerance=1e-4,
                 regular_term=1e-4,
                 logger=None):
        """Triangulator for points triangulation, based on jacobi optimization.

        Args:
            camera_parameters (List[FisheyeCameraParameter]):
                A list of Pinhole/FisheyeCameraParameter, or a list
                of paths to dumped Pinhole/FisheyeCameraParameters.
            maxIter_time (int):
                maximal iteration to optimize
            update_tolerance (float):
                indicator of convergent in optimization
            regular_term (float):
                regulat term
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(camera_parameters=camera_parameters, logger=logger)
        self.projs = None
        self.loss = None

        self.maxIter_time = maxIter_time
        self.update_tolerance = update_tolerance
        self.regular_term = regular_term

        self.logger = logger

        if len(self.camera_parameters) > 0:
            self._prepare_proj_mat(self.camera_parameters)

    def _solve(self, points, points_c):
        points = points.T
        convergent = False
        loss = 1e10
        pos = np.zeros(3, dtype=np.float32)

        if sum(points_c > 0) < 2:
            return pos, loss

        for iter_time in range(self.maxIter_time):
            if convergent:
                break
            ATA = self.regular_term * np.identity(3, dtype=np.float32)
            ATb = np.zeros(3, dtype=np.float32)
            for view in range(points.shape[1]):
                if points_c[view] > 0:
                    proj = self.projs[:, 4 * view:4 * view + 4]
                    xyz = np.matmul(proj, np.append(pos, 1))
                    jacobi = np.zeros((2, 3), dtype=np.float32)
                    jacobi = np.array([
                        1.0 / xyz[2], 0.0, -xyz[0] / (xyz[2] * xyz[2]), 0.0,
                        1.0 / xyz[2], -xyz[1] / (xyz[2] * xyz[2])
                    ],
                                      dtype=np.float32).reshape((2, 3))
                    jacobi = np.matmul(jacobi, proj[:, :3])
                    w = points_c[view]
                    ATA += w * np.matmul(jacobi.T, jacobi)
                    ATb += w * np.matmul(
                        jacobi.T, (points[:, view][:2] - xyz[:2] / xyz[2]))

            delta = np.linalg.solve(ATA, ATb)
            loss = np.linalg.norm(delta)
            if np.linalg.norm(delta) < self.update_tolerance:
                convergent = True
            else:
                pos += delta
        return pos, loss

    def triangulate(
            self,
            points: Union[np.ndarray, list, tuple],
            points_mask: Union[np.ndarray, list, tuple] = None) -> np.ndarray:

        points, points_mask = prepare_triangulate_input(
            camera_number=len(self.camera_parameters),
            points=points,
            points_mask=points_mask,
            logger=self.logger)

        points2d = points[..., :2].copy()
        input_points2d_shape = points2d.shape
        n_view = input_points2d_shape[0]
        points2d = points2d.reshape(n_view, -1, 2)
        points_mask = points_mask.reshape(n_view, -1, 1)
        ignored_indexes = np.where(points_mask != 1)
        points2d_c = points[..., 2].copy()
        points2d_c = points2d_c.reshape(n_view, -1, 1)
        points2d_c[ignored_indexes[0], ignored_indexes[1], :] = 0
        n_points = points2d.shape[1]
        self.loss = np.full(n_points, 10e9)
        points3d = []
        for point_id in range(n_points):
            pos, loss = self._solve(points2d[:, point_id],
                                    points2d_c[:, point_id])
            points3d.append(pos)
            self.loss[point_id] = loss
        points3d = np.array(points3d)

        output_points3d_shape = np.array(input_points2d_shape[1:])
        output_points3d_shape[-1] = 3
        points3d = points3d.reshape(*output_points3d_shape)
        return points3d

    def _prepare_proj_mat(self, camera_parameters) -> np.ndarray:
        projs = np.zeros((3, len(camera_parameters) * 4))
        for view in range(len(camera_parameters)):
            K = camera_parameters[view].intrinsic33()
            T = np.array(camera_parameters[view].get_extrinsic_t())
            R = np.array(camera_parameters[view].get_extrinsic_r())
            Proj = np.zeros((3, 4), dtype=np.float32)
            for i in range(3):
                for j in range(4):
                    Proj[i, j] = R[i, j] if j < 3 else T[i]
            projs[:, 4 * view:4 * view + 4] = np.matmul(K, Proj)
        self.projs = projs

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
        self._prepare_proj_mat(camera_parameters)
        super().set_cameras(camera_parameters=camera_parameters)
