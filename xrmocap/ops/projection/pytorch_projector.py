# yapf: disable
import logging
import torch
from typing import List, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.ops.projection.base_projector import BaseProjector
from xrprimer.utils.log_utils import get_logger

# yapf: enable


class PytorchProjector(BaseProjector):
    CAMERA_CONVENTION = 'opencv'
    CAMERA_WORLD2CAM = True

    def __init__(self,
                 camera_parameters: List[FisheyeCameraParameter],
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """PytorchProjector for points projection.

        Args:
            camera_parameters (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter.
            logger (Union[None, str, logging.Logger], optional):
                Defaults to None.
        """
        BaseProjector.__init__(self, camera_parameters)
        self.logger = get_logger(logger)

    def project(self,
                points: torch.Tensor,
                points_mask: torch.tensor = None) -> torch.Tensor:
        """Project points with self.camera_parameters.

        Args:
            points (torch.Tensor):
                points3d, in shape [n_point, 3].
            points_mask (torch.Tensor, optional):
                mask, in shape [n_point, 1].
                If points_mask[index] == 1, points[index] is valid
                for projection, else it is ignored.
                Defaults to None.

        Returns:
            torch.Tensor:
                points2d, in shape [n_view, n_point, 2].
        """
        points3d = points[..., :3].reshape(-1, 3).float()
        n_point = points3d.shape[0]
        n_view = len(self.camera_parameters)
        points2d = torch.zeros((n_view, n_point, 2), dtype=points3d.dtype)
        points_mask = points_mask.reshape(-1) \
            if points_mask is not None \
            else torch.ones(n_point, dtype=torch.uint8)
        valid_idxs = torch.where(points_mask == 1)
        mview_project_mat = self.prepare_project_mat()
        points3d = points3d[valid_idxs[0], :].T
        points3d_homo = torch.cat(
            (points3d, torch.ones(points3d.shape[-1]).reshape(1, -1)), dim=0)
        points2d_homo = mview_project_mat @ points3d_homo
        points2d_homo = points2d_homo.transpose(2, 1)
        proj_points2d = points2d_homo[..., :2] / (
            points2d_homo[..., 2:3] + 1e-5)
        points2d[:, valid_idxs[0], :] = proj_points2d
        return points2d

    def project_single_point(self, points: torch.Tensor) -> torch.Tensor:
        """Project a single point with self.camera_parameters.

        Args:
            points (torch.Tensor):
                points3d, in shape [3].

        Returns:
            torch.Tensor:
                points2d, in shape [n_view, 2].
        """
        points3d = points.reshape(1, 3)
        return torch.squeeze(self.project(points3d), dim=1)

    def prepare_project_mat(self):
        n_view = len(self.camera_parameters)
        mview_project_mat = torch.zeros((n_view, 3, 4))
        K = torch.zeros((n_view, 3, 3))
        RT = torch.zeros((n_view, 3, 4))
        for i, cam_param in enumerate(self.camera_parameters):
            K[i] = torch.tensor(cam_param.get_intrinsic(k_dim=3))
            RT[i] = torch.cat(
                (torch.tensor(cam_param.extrinsic_r),
                 torch.tensor(cam_param.extrinsic_t).unsqueeze(1)),
                dim=1)
            # compute projection matrix
            mview_project_mat[i] = K[i] @ RT[i]
        return mview_project_mat
