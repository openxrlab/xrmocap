import logging
import torch
from typing import TypeVar, Union

from xrmocap.model.loss.builder import build_loss
from xrmocap.transform.convention.keypoints_convention import convert_kps_mm
from .base_handler import BaseHandler, BaseInput

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_LimbLengthLoss = TypeVar('_LimbLengthLoss')


class Keypoint3dLimbLenInput(BaseInput):

    def __init__(
        self,
        keypoints3d: torch.Tensor,
        keypoints3d_convention: str = 'human_data',
        keypoints3d_conf: torch.Tensor = None,
        handler_key='keypoints3d_limb_len',
    ) -> None:
        """Construct an input instance for Keypoint3dLimbLenInput.

        Args:
            keypoints3d (torch.Tensor):
                3D keypoints in shape (batch_size, n_keypoints, 3).
            keypoints3d_convention (torch.Tensor):
                Convention name of the 3D keypoints.
                Defaults to 'human_data'.
            keypoints3d_conf (torch.Tensor, optional):
                3D keypoints confidence in shape (batch_size, n_keypoints).
                Defaults to None, which will be regarded as torch.ones().
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'keypoints3d_limb_len'.
        """
        self.keypoints3d = keypoints3d
        self.keypoints3d_convention = keypoints3d_convention
        self.keypoints3d_conf = keypoints3d_conf \
            if keypoints3d_conf is not None else \
            torch.ones_like(keypoints3d[:, :, 0])
        super().__init__(handler_key=handler_key)

    def get_batch_size(self) -> int:
        """Get batch size of the input.

        Returns:
            int: batch_size
        """
        return int(self.keypoints3d.shape[0])


class Keypoint3dLimbLenHandler(BaseHandler):

    def __init__(self,
                 loss: Union[_LimbLengthLoss, dict],
                 handler_key='keypoints3d_limb_len',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a Keypoint3dMSEHandler instance compute smpl(x/xd)
        parameters and BaseInput, return a loss Tensor.

        Args:
            loss (Union[LimbLengthLoss, dict]):
                An instance of LimbLengthLoss, or a config dict of
                LimbLengthLoss.
            keypoint_approximate (bool, optional):
                Whether to allow approximate mapping. Defaults to False.
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'keypoints3d_limb_len'.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            TypeError: loss is neither a torch.nn.Module nor a dict.
        """
        super().__init__(handler_key=handler_key, device=device, logger=logger)
        if isinstance(loss, dict):
            self.loss = build_loss(loss)
        elif isinstance(loss, torch.nn.Module):
            self.loss = loss
        else:
            self.logger.error('Type of loss is not correct.\n' +
                              f'Type: {type(loss)}.')
            raise TypeError
        self.loss = self.loss.to(self.device)

    def get_loss_weight(self) -> float:
        """Get the weight value of this loss handler.

        Returns:
            float: Weight value.
        """
        loss_weight = self.loss.loss_weight
        return float(loss_weight)

    def __call__(self,
                 related_input: Keypoint3dLimbLenInput,
                 model_joints: torch.Tensor,
                 model_joints_convention: str,
                 loss_weight_override: float = None,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs: dict) -> torch.Tensor:
        """Taking Keypoint3dLimbLenInput and smpl(x/xd) parameters, compute
        loss and return a Tensor.

        Args:

            related_input (Keypoint3dLimbLenInput):
                An instance of Keypoint3dLimbLenInput, having the same
                key as self.
            model_joints (torch.Tensor):
                Joints from body_model.
            model_joints_convention (str):
                Convention name of the model_joints.
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
        target_keypoints3d = related_input.keypoints3d
        target_keypoints_convention = related_input.keypoints3d_convention
        target_keypoints3d_conf = related_input.keypoints3d_conf
        joints, joint_mask = convert_kps_mm(
            keypoints=model_joints,
            src=model_joints_convention,
            dst=self.loss.convention)
        batch_size = related_input.get_batch_size()
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)
        target_keypoints3d, target_keypoints3d_conf = convert_kps_mm(
            keypoints=target_keypoints3d,
            src=target_keypoints_convention,
            dst=self.loss.convention,
            mask=target_keypoints3d_conf[0])
        target_keypoints3d_conf = target_keypoints3d_conf.reshape(
            1, -1).expand(batch_size, -1)
        limb_len_loss = self.loss(
            pred=joints,
            pred_conf=joint_mask,
            target=target_keypoints3d,
            target_conf=target_keypoints3d_conf,
            loss_weight_override=loss_weight_override,
            reduction_override=reduction_override)
        return limb_len_loss
