import logging
import torch
from pytorch3d.renderer import cameras
from typing import TypeVar, Union

from xrmocap.model.loss.builder import build_loss
from xrmocap.transform.convention.keypoints_convention import convert_kps_mm
from .base_handler import BaseHandler, BaseInput

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_KeypointMSELoss = TypeVar('_KeypointMSELoss')


class MultiviewKeypoint2dMSEInput(BaseInput):

    def __init__(
        self,
        cameras: cameras,
        keypoints2d: torch.Tensor,
        keypoints2d_convention: str = 'human_data',
        keypoints2d_conf: torch.Tensor = None,
        keypoints_weight: torch.Tensor = None,
        handler_key='mview_keypoints2d_mse',
    ) -> None:
        """Construct an input instance for Keypoint3dMSEInput.

        Args:
            cameras (pytorch3d.renderer.cameras):
                Pytorch 3d cameras, len(cameras) = batch_size * n_view.
            keypoints2d (torch.Tensor):
                2D keypoints in shape (batch_size, n_view, n_keypoints, 2).
            keypoints2d_convention (torch.Tensor):
                Convention name of the 2D keypoints.
                Defaults to 'human_data'.
            keypoints2d_conf (torch.Tensor, optional):
                2D keypoints confidence in shape
                (batch_size, n_view, n_keypoints).
                Defaults to None, which will be regarded as torch.ones().
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'mview_keypoints2d_mse'.
        """
        self.keypoints2d = keypoints2d
        self.cameras = cameras
        self.keypoints2d_convention = keypoints2d_convention
        self.keypoints2d_conf = keypoints2d_conf \
            if keypoints2d_conf is not None else \
            torch.ones_like(keypoints2d[:, :, :, 0])
        super().__init__(handler_key=handler_key)

    def get_batch_size(self) -> int:
        """Get batch size of the input.

        Returns:
            int: batch_size
        """
        return int(self.keypoints2d.shape[0])


class MultiviewKeypoint2dMSEHandler(BaseHandler):

    def __init__(self,
                 mse_loss: Union[_KeypointMSELoss, dict],
                 keypoint_approximate: bool = False,
                 handler_key='mview_keypoints2d_mse',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a MultiviewKeypoint2dMSEHandler instance compute
        smpl(x/xd) parameters and BaseInput, return a loss Tensor.

        Args:
            mse_loss (Union[KeypointMSELoss, dict]):
                An instance of KeypointMSELoss, or a config dict of
                KeypointMSELoss.
            keypoint_approximate (bool, optional):
                Whether to allow approximate mapping. Defaults to False.
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'mview_keypoints2d_mse'.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            TypeError: mse_loss is neither a torch.nn.Module nor a dict.
        """
        super().__init__(handler_key=handler_key, device=device, logger=logger)
        if isinstance(mse_loss, dict):
            self.mse_loss = build_loss(mse_loss)
        elif isinstance(mse_loss, torch.nn.Module):
            self.mse_loss = mse_loss
        else:
            self.logger.error('Type of mse_loss is not correct.\n' +
                              f'Type: {type(mse_loss)}.')
            raise TypeError
        self.keypoint_approximate = keypoint_approximate
        self.mse_loss = self.mse_loss.to(self.device)

    def get_loss_weight(self) -> float:
        """Get the weight value of this loss handler.

        Returns:
            float: Weight value.
        """
        loss_weight = self.mse_loss.loss_weight
        return float(loss_weight)

    def __call__(self,
                 related_input: MultiviewKeypoint2dMSEInput,
                 model_joints: torch.Tensor,
                 model_joints_convention: str,
                 model_joints_weights: torch.Tensor,
                 loss_weight_override: float = None,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs: dict) -> torch.Tensor:
        """Taking MultiviewKeypoint2dMSEInput and smpl(x/xd) parameters,
        compute loss and return a Tensor.

        Args:

            related_input (MultiviewKeypoint2dMSEInput):
                An instance of MultiviewKeypoint2dMSEInput,
                having the same key as self.
            model_joints (torch.Tensor):
                Joints from body_model.
            model_joints_convention (str):
                Convention name of the model_joints.
            model_joints_weights (torch.Tensor):
                If given, each joint has its own weight, in shape (K, ).
                Defaults to None.
            loss_weight_override (float, optional):
                Override the global weight of this loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional):
                Override the reduction method of this loss.
                Defaults to None.
            kwargs (dict):
                Redundant smpl(x/d) keyword arguments to be
                ignored.

        Raises:
            ValueError:
                Value of reduction_override is not in
                [\'mean\', \'sum\', \'none\'].

        Returns:
            torch.Tensor:
                A Tensor of loss result.
        """
        if reduction_override not in ['mean', 'sum', 'none', None]:
            self.logger.error('Value of reduction_override' +
                              'is not in [\'mean\', \'sum\', \'none\'].\n' +
                              f'reduction_override: {reduction_override}.')
            raise ValueError
        target_keypoints2d = related_input.keypoints2d
        b, v, k, n = target_keypoints2d.shape
        target_keypoints2d = target_keypoints2d.reshape(b, -1, n)
        target_keypoints_convention = related_input.keypoints2d_convention
        target_keypoints2d_conf = related_input.keypoints2d_conf
        cameras = related_input.cameras
        projected_joints_xyd = cameras.transform_points_screen(
            points=model_joints)
        projected_joints = projected_joints_xyd[..., :2]
        # todo: check shape and dim
        # todo: make sure keypoints in range (-1, 1)
        projected_joints, joint_mask = convert_kps_mm(
            keypoints=projected_joints,
            src=model_joints_convention,
            dst=target_keypoints_convention,
            approximate=self.keypoint_approximate)
        batch_size = related_input.get_batch_size()
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)
        keypoints3d_loss = self.mse_loss(
            pred=projected_joints,
            pred_conf=joint_mask,
            target=target_keypoints2d,
            target_conf=target_keypoints2d_conf,
            keypoint_weight=model_joints_weights,
            loss_weight_override=loss_weight_override,
            reduction_override=reduction_override)
        return keypoints3d_loss
