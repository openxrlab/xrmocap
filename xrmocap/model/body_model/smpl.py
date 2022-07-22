# yapf: disable
import logging
import numpy as np
import torch
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.convention.keypoints_convention import (
    get_keypoints_factory,
)

# yapf: enable


class SMPL(_SMPL):
    NUM_VERTS = 6890
    NUM_FACES = 13776

    def __init__(self,
                 *args,
                 keypoint_convention: str = 'smpl_45',
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 logger: Union[None, str, logging.Logger] = None,
                 **kwargs) -> None:
        """Extension of the official SMPL implementation.

        Args:
            *args:
                extra arguments for SMPL initialization.
            keypoint_convention (str, optional):
                Source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Defaults to 'smpl_45'.
            joints_regressor (str, optional):
                Path to joint regressor. Should be an npy file.
                If provided, replaces the official J_regressor of SMPL.
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
                extra keyword arguments for SMPL initialization.
        """
        super(SMPL, self).__init__(*args, **kwargs)
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
            *args: extra arguments for SMPL
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPL

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smpl_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smpl_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        output = dict(
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            betas=smpl_output.betas)

        if return_verts:
            output['vertices'] = smpl_output.vertices
        if return_full_pose:
            output['full_pose'] = smpl_output.full_pose

        return output

    def get_joint_number(self) -> int:
        """Get the number of joint from this body_model.

        Returns:
            int: Number of joint(keypoint).
        """
        return len(get_keypoints_factory()[self.keypoint_convention])
