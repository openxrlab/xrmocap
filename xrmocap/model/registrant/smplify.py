import logging
import numpy as np
import prettytable
import torch
from mmcv.runner import build_optimizer
from typing import List, Union

from xrmocap.model.body_model.builder import build_body_model
from xrmocap.transform.convention.keypoints_convention import (  # noqa:E501
    get_keypoint_idx, get_keypoint_idxs_by_part,
)
from xrmocap.utils.log_utils import get_logger
from .handler.base_handler import BaseHandler, BaseInput
from .handler.builder import build_handler
from .optimizable_parameters import OptimizableParameters

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class SMPLify(object):
    """Re-implementation of SMPLify with extended features."""

    def __init__(self,
                 body_model: Union[dict, torch.nn.Module],
                 stages: dict,
                 optimizer: dict,
                 handlers: List[Union[dict, BaseHandler]],
                 num_epochs: int = 1,
                 use_one_betas_per_video: bool = False,
                 ignore_keypoints: List[str] = None,
                 device: Union[torch.device, str] = 'cuda',
                 verbose: bool = False,
                 info_level: Literal['stage', 'step'] = 'step',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Re-implementation of SMPLify with extended features.

        Args:
            body_model (Union[dict, torch.nn.Module]):
                An instance of SMPL body_model or a config dict
                of SMPL body_model.
            stages (dict):
                A config dict of registration stages.
            optimizer (dict):
                A config dict of optimizer.
            handlers (List[Union[dict, BaseHandler]]):
                A list of handlers, each element is an instance of
                subclass of BaseHandler, or a config dict of handler.
            num_epochs (int, optional):
                Number of epochs. Defaults to 1.
            use_one_betas_per_video (bool, optional):
                Whether to use the same beta parameters
                for all frames in a single video sequence.
                Defaults to False.
            ignore_keypoints (List[str], optional):
                A list of keypoint names to ignore in keypoint
                loss computation. Defaults to None.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            verbose (bool, optional):
                Whether to print individual losses during registration.
                Defaults to False.
            info_level (Literal['stage', 'step']):
                Whether to print information every stage or every step.
                Defaults to stage.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            TypeError: type of body_model should be
            either dict or torch.nn.Module.
        """
        self.logger = get_logger(logger)
        self.info_level = info_level
        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.device = device
        self.stage_config = stages
        self.optimizer = optimizer

        # initialize body model
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            self.logger.error('body_model should be either dict or ' +
                              f'torch.nn.Module, but got {type(body_model)}.')
            raise TypeError

        self.ignore_keypoints = ignore_keypoints \
            if ignore_keypoints is not None \
            else []
        self.verbose = verbose
        self.loss_handlers = []
        for handler_arg in handlers:
            if isinstance(handler_arg, dict):
                handler_arg['device'] = self.device
                handler = build_handler(handler_arg)
            else:
                handler = handler_arg
            self.loss_handlers.append(handler)
        self.__set_keypoint_idxs__()
        self.__stage_kwargs_warned__ = False

    def __call__(self,
                 input_list: List[BaseInput],
                 init_global_orient: torch.Tensor = None,
                 init_transl: torch.Tensor = None,
                 init_body_pose: torch.Tensor = None,
                 init_betas: torch.Tensor = None,
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        """Run registration.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            init_global_orient (torch.Tensor, optional):
                Initial global_orient of shape (B, 3).
                Defaults to None.
            init_transl (torch.Tensor, optional):
                Initial transl of shape (B, 3). Defaults to None.
            init_body_pose (torch.Tensor, optional):
                Initial body_pose of shape (B, 69). Defaults to None.
            init_betas (torch.Tensor, optional):
                Initial betas of shape (B, D). Defaults to None.
            return_verts (bool, optional):
                Whether to return vertices. Defaults to False.
            return_joints (bool, optional):
                Whether to return joints. Defaults to False.
            return_full_pose (bool, optional):
                Whether to return full pose. Defaults to False.
            return_losses (bool, optional):
                Whether to return loss dict. Defaults to False.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Returns:
            dict:
                A dictionary that includes body model parameters,
                and optional attributes such as vertices and joints.
        """
        batch_size = None
        for input_instance in input_list:
            tmp_batch_size = input_instance.get_batch_size()
            if batch_size is None:
                batch_size = tmp_batch_size
            elif batch_size != tmp_batch_size:
                self.logger.error('Batch size varies among input_list.\n' +
                                  f'Target batch size: {batch_size}\n' +
                                  f'{input_instance.handler_key} batch size:' +
                                  f' {tmp_batch_size}')
                raise ValueError

        global_orient = self.__match_init_batch_size__(
            init_global_orient, self.body_model.global_orient, batch_size)
        transl = self.__match_init_batch_size__(init_transl,
                                                self.body_model.transl,
                                                batch_size)
        body_pose = self.__match_init_batch_size__(init_body_pose,
                                                   self.body_model.body_pose,
                                                   batch_size)
        if init_betas is None and self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self.__match_init_batch_size__(init_betas,
                                                   self.body_model.betas,
                                                   batch_size)

        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                if self.verbose:
                    self.logger.info(f'epoch {i}, stage {stage_idx}')
                self.__optimize_stage__(
                    input_list=input_list,
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    **stage_config,
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

        if return_verts or return_joints or \
                return_full_pose or return_losses:
            eval_ret = self.evaluate(
                input_list=input_list,
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                return_verts=return_verts,
                return_full_pose=return_full_pose,
                return_joints=return_joints,
                reduction_override='none'  # sample-wise loss
            )

            if return_verts:
                ret['vertices'] = eval_ret['vertices']
            if return_joints:
                ret['joints'] = eval_ret['joints']
            if return_full_pose:
                ret['full_pose'] = eval_ret['full_pose']
            if return_losses:
                for k in eval_ret.keys():
                    if 'loss' in k:
                        ret[k] = eval_ret[k]

        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()

        return ret

    def __optimize_stage__(self,
                           input_list: List[BaseInput],
                           betas: torch.Tensor,
                           body_pose: torch.Tensor,
                           global_orient: torch.Tensor,
                           transl: torch.Tensor,
                           fit_global_orient: bool = True,
                           fit_transl: bool = True,
                           fit_body_pose: bool = True,
                           fit_betas: bool = True,
                           use_shoulder_hip_only: bool = False,
                           body_weight: float = 1.0,
                           num_iter: int = 1,
                           ftol: float = 1e-4,
                           **kwargs) -> None:
        """Optimize a stage of body model parameters according to
        configuration.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            betas (torch.Tensor):
                Shape (B, D).
            body_pose (torch.Tensor):
                Shape (B, 69).
            global_orient (torch.Tensor):
                Shape (B, 3).
            transl (torch.Tensor):
                Shape (B, 3).
            fit_global_orient (bool, optional):
                Whether to optimize global_orient. Defaults to True.
            fit_transl (bool, optional):
                Whether to optimize transl. Defaults to True.
            fit_body_pose (bool, optional):
                Whether to optimize body_pose. Defaults to True.
            fit_betas (bool, optional):
                Whether to optimize betas. Defaults to True.
            use_shoulder_hip_only (bool, optional):
                Keypoints weight argument.
                Whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
                Defaults to False.
            body_weight (float, optional):
                Keypoints weight argument.
                Weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            num_iter (int, optional):
                Number of iterations. Defaults to 1.
            ftol (float, optional):
                Defaults to 1e-4.

        kwargs:
            Stage control keyword arguments, including override weights
            of each loss handler.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        """
        parameters = OptimizableParameters()
        parameters.add_param(
            key='global_orient',
            param=global_orient,
            fit_param=fit_global_orient)
        parameters.add_param(key='transl', param=transl, fit_param=fit_transl)
        parameters.add_param(
            key='body_pose', param=body_pose, fit_param=fit_body_pose)
        parameters.add_param(key='betas', param=betas, fit_param=fit_betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        pre_loss = None
        for iter_idx in range(num_iter):

            def closure():
                optimizer.zero_grad()
                betas_video = self.__expand_betas__(body_pose.shape[0], betas)

                loss_dict = self.evaluate(
                    input_list=input_list,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
                    transl=transl,
                    use_shoulder_hip_only=use_shoulder_hip_only,
                    body_weight=body_weight,
                    **kwargs)

                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if iter_idx > 0 and pre_loss is not None and ftol > 0:
                loss_rel_change = self.__compute_relative_change__(
                    pre_loss, loss.item())
                if loss_rel_change < ftol:
                    if self.verbose:
                        self.logger.info(
                            f'[ftol={ftol}] Early stop at {iter_idx} iter!')
                    break
            pre_loss = loss.item()

        # log stage information
        if self.verbose:
            with torch.no_grad():
                betas_video = self.__expand_betas__(body_pose.shape[0], betas)
                losses = self.evaluate(
                    input_list=input_list,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
                    transl=transl,
                    use_shoulder_hip_only=use_shoulder_hip_only,
                    body_weight=body_weight,
                    **kwargs)
            table = prettytable.PrettyTable()
            table.field_names = ['Loss name', 'Loss value']
            for key, value in losses.items():
                if isinstance(value, float) or \
                        isinstance(value, int) or \
                        len(value.shape) == 0:
                    table.add_row([key, f'{value:.6f}'])
                else:
                    table.add_row([key, 'Not a scalar'])
            table = '\n' + table.get_string()
            self.logger.info(table)

    def evaluate(self,
                 input_list: List[BaseInput],
                 betas: torch.Tensor = None,
                 body_pose: torch.Tensor = None,
                 global_orient: torch.Tensor = None,
                 transl: torch.Tensor = None,
                 return_verts: bool = False,
                 return_full_pose: bool = False,
                 return_joints: bool = False,
                 use_shoulder_hip_only: bool = False,
                 body_weight: float = 1.0,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs) -> dict:
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            betas (torch.Tensor):
                Shape (B, D).
            body_pose (torch.Tensor):
                Shape (B, 69).
            global_orient (torch.Tensor):
                Shape (B, 3).
            transl (torch.Tensor):
                Shape (B, 3).
            return_verts (bool, optional):
                Whether to return vertices. Defaults to False.
            return_joints (bool, optional):
                Whether to return joints. Defaults to False.
            return_full_pose (bool, optional):
                Whether to return full pose. Defaults to False.
            return_losses (bool, optional):
                Whether to return loss dict. Defaults to False.
            use_shoulder_hip_only (bool, optional):
                Keypoints weight argument.
                Whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
                Defaults to False.
            body_weight (float, optional):
                Keypoints weight argument.
                Weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            reduction_override (Literal['mean', 'sum', 'none'], optional):
                The reduction method. If given, it will
                override the original reduction method of the loss.
                Defaults to None.

        kwargs:
            Stage control keyword arguments, including override weights
            of eache loss handler.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Returns:
            dict:
                A dictionary that includes body model parameters,
                and optional attributes such as vertices and joints.
        """
        ret = {}

        # check if verts are essantial
        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=return_verts,
            return_full_pose=return_full_pose)

        model_joints = body_model_output['joints']
        model_joints_convention = self.body_model.keypoint_convention
        model_joints_weights = self.get_keypoint_weight(
            use_shoulder_hip_only=use_shoulder_hip_only,
            body_weight=body_weight)
        model_vertices = body_model_output.get('vertices', None)

        loss_dict = self.__compute_loss__(
            input_list=input_list,
            model_joints=model_joints,
            model_joints_convention=model_joints_convention,
            model_joints_weights=model_joints_weights,
            model_vertices=model_vertices,
            reduction_override=reduction_override,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            **kwargs)
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints

        return ret

    def __compute_loss__(self,
                         input_list: List[BaseInput],
                         model_joints: torch.Tensor = None,
                         model_joints_convention: str = '',
                         model_joints_weights: torch.Tensor = None,
                         model_vertices: torch.Tensor = None,
                         betas: torch.Tensor = None,
                         body_pose: torch.Tensor = None,
                         global_orient: torch.Tensor = None,
                         transl: torch.Tensor = None,
                         reduction_override: Literal['mean', 'sum',
                                                     'none'] = None,
                         **kwargs) -> dict:
        """Loss computation.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            model_joints (torch.Tensor):
                Output joints from self.body_model, of shape (B, K, 3).
                Defaults to None.
            model_joints_convention (str):
                Convention name of model_joints. Defaults to ''.
            model_joints_weights (torch.Tensor):
                If given, each joint has its own weight, in shape (K, ).
                Defaults to None.
            model_vertices (torch.Tensor):
                Output joints from self.body_model. Defaults to None.
            betas (torch.Tensor):
                Shape (B, D).
            body_pose (torch.Tensor):
                Shape (B, 69).
            global_orient (torch.Tensor):
                Shape (B, 3).
            transl (torch.Tensor):
                Shape (B, 3).
            reduction_override (Literal['mean', 'sum', 'none'], optional):
                The reduction method. If given, it will
                override the original reduction method of the loss.
                Defaults to None.

        kwargs:
            Stage control keyword arguments, including override weights
            of each loss handler.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Returns:
            dict: A dict that contains all losses.
        """
        if self.use_one_betas_per_video:
            betas = betas.repeat(global_orient.shape[0], 1)
        if model_joints_weights is None and \
                model_joints is not None:
            model_joints_weights = torch.ones_like(model_joints[0, :, 0])
        init_handler_input = {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
            'model_vertices': model_vertices,
            'model_joints': model_joints,
            'model_joints_convention': model_joints_convention,
            'model_joints_weights': model_joints_weights,
        }
        losses = {}
        # backup kwargs for pop
        kwargs = kwargs.copy()
        for handler in self.loss_handlers:
            handler_key = handler.handler_key
            # Get args from stage controller.
            # If used, pop it.
            loss_weight_override = kwargs.pop(f'{handler_key}_weight', None)
            reduction_override = reduction_override \
                if reduction_override is not None \
                else kwargs.pop(
                    f'{handler_key}_reduction', None)
            if self.__skip_handler__(
                    loss_handler=handler,
                    loss_weight_override=loss_weight_override):
                continue
            handler_input = init_handler_input.copy()
            handler_input.update({
                'loss_weight_override': loss_weight_override,
                'reduction_override': reduction_override,
            })
            loss_tensor = None
            # e.g. shape prior loss
            if not handler.requires_input():
                loss_tensor = handler(**handler_input)
            # e.g. keypoints 3d mes loss
            else:
                for input_inst in input_list:
                    if input_inst.handler_key == handler_key:
                        handler_input['related_input'] = input_inst
                        loss_tensor = handler(**handler_input)
            # if loss computed, record it in losses
            if loss_tensor is not None:
                losses[handler_key] = loss_tensor

        total_loss = 0
        for key, loss in losses.items():
            if loss.ndim == 3:
                total_loss = total_loss + loss.sum(dim=(2, 1))
            elif loss.ndim == 2:
                total_loss = total_loss + loss.sum(dim=-1)
            else:
                total_loss = total_loss + loss
        losses['total_loss'] = total_loss

        # warn once if there's item still in popped kwargs
        if not self.__stage_kwargs_warned__ and \
                len(kwargs) > 0:
            warn_msg = 'Args below do not match any loss handler:\n'
            table = prettytable.PrettyTable()
            table.field_names = ['Arg key', 'Arg value']
            for key, value in kwargs.items():
                table.add_row([key, value])
            warn_msg = warn_msg + table.get_string()
            self.logger.warning(warn_msg)
            self.__stage_kwargs_warned__ = True

        if self.verbose and self.info_level == 'step':
            table = prettytable.PrettyTable()
            table.field_names = ['Loss name', 'Loss value']
            for key, value in losses.items():
                if isinstance(value, float) or \
                        isinstance(value, int) or \
                        len(value.shape) == 0:
                    table.add_row([key, f'{value:.6f}'])
                else:
                    table.add_row([key, 'Not a scalar'])
            table = '\n' + table.get_string()
            self.logger.info(table)

        return losses

    def __match_init_batch_size__(self, init_param: torch.Tensor,
                                  init_param_body_model: torch.Tensor,
                                  batch_size: int) -> torch.Tensor:
        """A helper function to ensure body model parameters have the same
        batch size as the input keypoints.

        Args:
            init_param: input initial body model parameters, may be None
            init_param_body_model: initial body model parameters from the
                body model
            batch_size: batch size of keypoints

        Returns:
            param: body model parameters with batch size aligned
        """

        # param takes init values
        param = init_param.detach().clone() \
            if init_param is not None \
            else init_param_body_model.detach().clone()

        # expand batch dimension to match batch size
        param_batch_size = param.shape[0]
        if param_batch_size != batch_size:
            if param_batch_size == 1:
                param = param.repeat(batch_size, *[1] * (param.ndim - 1))
            else:
                self.logger.error(
                    'Init param does not match the batch size of '
                    'keypoints, and is not 1.')
                raise ValueError

        # shape check
        if param.shape[0] != batch_size or \
                param.shape[1:] != init_param_body_model.shape[1:]:
            self.logger.error(f'Shape mismatch: {param.shape} vs' +
                              f' {init_param_body_model.shape}')
            raise ValueError
        return param

    def __set_keypoint_idxs__(self) -> None:
        """Set keypoint indices to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.

        Returns:
            None
        """
        convention = self.body_model.keypoint_convention

        # obtain ignore keypoint indices
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(
                    keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indices
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)
        hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [
            *shoulder_keypoint_idxs, *hip_keypoint_idxs
        ]

    def get_keypoint_weight(self,
                            use_shoulder_hip_only: bool = False,
                            body_weight: float = 1.0) -> torch.Tensor:
        """Get per keypoint weight.

        Args:
            use_shoulder_hip_only (bool, optional):
                Whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
                Defaults to False.
            body_weight (float, optional):
                Weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.

        Returns:
            torch.Tensor: Per keypoint weight tensor of shape (K).
        """
        num_keypoint = self.body_model.get_joint_number()

        if use_shoulder_hip_only:
            weight = torch.zeros([num_keypoint]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            weight = weight * body_weight
        else:
            weight = torch.ones([num_keypoint]).to(self.device)
            weight = weight * body_weight

        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight

    def __expand_betas__(self, batch_size, betas):
        """A helper function to expand the betas's first dim to match batch
        size such that the same beta parameters can be used for all frames in a
        video sequence.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            batch_size: batch size
            betas: shape (B, D)

        Returns:
            betas_video: expanded betas
        """
        # no expansion needed
        if batch_size == betas.shape[0]:
            return betas

        # first dim is 1
        else:
            feat_dim = betas.shape[-1]
            betas_video = betas.view(1, feat_dim).expand(batch_size, feat_dim)

        return betas_video

    def __check_verts_requirement__(self, input_list: List[BaseInput],
                                    **kwargs) -> bool:
        """Check whether vertices are required by loss handlers.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.

        kwargs:
            Stage control keyword arguments, including override weights
            of each loss handler.

        Returns:
            bool: _description_
        """
        raise NotImplementedError

    @staticmethod
    def __compute_relative_change__(pre_v, cur_v):
        """Compute relative loss change. If relative change is small enough, we
        can apply early stop to accelerate the optimization. (1) When one of
        the value is larger than 1, we calculate the relative change by diving
        their max value. (2) When both values are smaller than 1, it degrades
        to absolute change. Intuitively, if two values are small and close,
        dividing the difference by the max value may yield a large value.

        Args:
            pre_v: previous value
            cur_v: current value

        Returns:
            float: relative change
        """
        return np.abs(pre_v - cur_v) / max([np.abs(pre_v), np.abs(cur_v), 1])

    @staticmethod
    def __skip_handler__(loss_handler: BaseHandler,
                         loss_weight_override: float) -> bool:
        """Whether to skip loss computation. If loss is None, it will directly
        skip the loss to avoid RuntimeError. If loss is not None, the table
        below shows the return value. If the return value is True, it means the
        computation of loss can be skipped. As the result is 0 even if it is
        calculated, we can skip it to save computational cost.

        | loss.loss_weight | loss_weight_override | returns |
        | ---------------- | -------------------- | ------- |
        |      == 0        |         None         |   True  |
        |      != 0        |         None         |   False |
        |      == 0        |         == 0         |   True  |
        |      != 0        |         == 0         |   True  |
        |      == 0        |         != 0         |   False |
        |      != 0        |         != 0         |   False |

        Args:
            loss_handler(BaseHandler):
                loss_handler is an instance of subclass of BaseHandler.
                loss_handler.get_loss_weight() is assigned
                when loss is initialized.
            loss_weight_override:
                loss_weight_override used to override init loss_weight.

        Returns:
            bool: True means skipping loss computation, and vice versa.
        """
        if loss_handler is None or \
                (loss_handler.get_loss_weight() == 0 and
                 loss_weight_override is None) or \
                loss_weight_override == 0:
            return True
        else:
            return False
