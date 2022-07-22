# yapf: disable
import logging
import numpy as np
import torch
from smplx import SMPLX as _SMPLX
from smplx.lbs import vertices2joints
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.convention.keypoints_convention import (
    get_keypoints_factory,
)

# yapf: enable


class SMPLX(_SMPLX):
    NUM_VERTS = 10475
    NUM_FACES = 20908

    def __init__(self,
                 *args,
                 use_face_contour: bool = True,
                 use_pca: bool = False,
                 flat_hand_mean: bool = True,
                 keypoint_convention: str = 'smplx',
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 logger: Union[None, str, logging.Logger] = None,
                 **kwargs) -> None:
        """Extension of the official SMPLX implementation.

        Args:
            *args:
                extra arguments for SMPLX initialization.
            use_face_contour (bool, optional):
                Whether to compute the keypoints that form the facial contour.
                If selected, there will be 144 keypoints.
                Defaults to True.
            use_pca (bool, optional):
                Whether to use pca for hand poses.
                Defaults to False.
            flat_hand_mean (bool, optional):
                If False, then the pose of the hand is initialized to False.
                Defaults to True.
            keypoint_convention (str, optional):
                Source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Defaults to 'smplx'.
            joints_regressor (str, optional):
                Path to joint regressor. Should be an npy file.
                If provided, replaces the official J_regressor of SMPLX.
                Defaults to None.
            extra_joints_regressor (str, optional):
                Path to extra joint regressor. Should be
                an npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
            **kwargs:
                extra keyword arguments for SMPLX initialization.
        """
        super(SMPLX, self).__init__(
            *args,
            use_face_contour=use_face_contour,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            **kwargs)
        self.logger = get_logger(logger)
        self.keypoint_convention = keypoint_convention
        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPLX
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPLX

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smplx_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smplx_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smplx_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose
        return output

    def get_joint_number(self) -> int:
        """Get the number of joint from this body_model.

        Returns:
            int: Number of joint(keypoint).
        """
        return len(get_keypoints_factory()[self.keypoint_convention])
