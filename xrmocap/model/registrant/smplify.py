# yapf: disable
import logging
import numpy as np
import prettytable
import torch
from mmcv.runner import build_optimizer
from mmcv.runner.hooks import Hook
from typing import List, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.core.hook.smplify_hook.builder import (
    SMPLifyBaseHook, build_smplify_hook,
)
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.model.loss.mapping import LOSS_MAPPING
from xrmocap.transform.convention.keypoints_convention import (  # noqa:E501
    get_keypoint_idx, get_keypoint_idxs_by_part,
)
from .handler.base_handler import BaseHandler, BaseInput
from .handler.builder import build_handler
from .optimizable_parameters import OptimizableParameters

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
# yapf: enable


class SMPLify(object):
    """Re-implementation of SMPLify with extended features."""
    OPTIM_PARAM = ['global_orient', 'transl', 'body_pose', 'betas']

    def __init__(self,
                 body_model: Union[dict, torch.nn.Module],
                 stages: dict,
                 optimizer: dict,
                 handlers: List[Union[dict, BaseHandler]],
                 n_epochs: int = 1,
                 use_one_betas_per_video: bool = False,
                 ignore_keypoints: List[str] = None,
                 device: Union[torch.device, str] = 'cuda',
                 hooks: List[Union[dict, SMPLifyBaseHook]] = [],
                 verbose: bool = False,
                 info_level: Literal['stage', 'step'] = 'step',
                 grad_clip: float = 1.0,
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
            n_epochs (int, optional):
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
            hooks (List[Union[dict, SMPLifyBaseHook]], optional):
                A list of hooks, each element is an instance of
                subclass of SMPLifyBaseHook, or a config dict of hook.
                Defaults to an empty list.
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
        self.n_epochs = n_epochs
        self.device = device
        self.stage_config = stages
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.hooks = []
        self.individual_optimizer = False

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
        # init handlers
        for handler in handlers:
            if isinstance(handler, dict):
                handler['device'] = self.device
                handler = build_handler(handler)
            self.loss_handlers.append(handler)
        # init hooks
        for hook in hooks:
            if isinstance(hook, dict):
                hook = build_smplify_hook(hook)
            self.register_hook(hook)
        self.__set_keypoint_indexes__()
        self.__stage_kwargs_warned__ = False

    def register_hook(self, hook: SMPLifyBaseHook):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue.

        Args:
            hook (SMPLifyBaseHook):
                The hook to be registered.
        """
        assert isinstance(hook, Hook)
        hook.priority = 'NORMAL'
        self.hooks.append(hook)

    def call_hook(self, fn_name: str, **kwargs):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_optimize".

        kwargs:
            Keyword args required by the hook.
        """
        for hook in self.hooks:
            getattr(hook, fn_name)(self, **kwargs)

    def __prepare_optimizable_parameters__(self, init_dict: dict,
                                           batch_size: int) -> dict:
        """Prepare optimizable parameters in batch for registrant. If some of
        the parameters can be found in init_dict, use them for initialization.

        Args:
            init_dict (dict):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
            batch_size (int)

        Returns:
            dict:
                A dict of optimizable parameters, whose keys are
                self.__class__.OPTIM_PARAM and values are
                Tensors in batch.
        """
        ret_dict = {}
        for key in self.__class__.OPTIM_PARAM:
            if key in init_dict:
                init_param = init_dict[key].to(self.device)
            else:
                init_param = None
            ret_param = self.__match_init_batch_size__(
                init_param=init_param,
                default_param=getattr(self.body_model, key),
                batch_size=batch_size)
            ret_dict[key] = ret_param
        if self.use_one_betas_per_video and 'betas' not in init_dict:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
            ret_dict['betas'] = betas
        return ret_dict

    def __call__(self,
                 input_list: List[BaseInput],
                 init_param_dict: dict = {},
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        """Run registration.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            init_param_dict (dict, optional):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
                Defaults to an empty dict.
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

        optim_param = self.__prepare_optimizable_parameters__(
            init_param_dict, batch_size)

        hook_kwargs = dict(input_list=input_list, optim_param=optim_param)
        self.call_hook('before_optimize', **hook_kwargs)

        for i in range(self.n_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                self.__optimize_stage__(
                    input_list=input_list,
                    optim_param=optim_param,
                    epoch_idx=i,
                    stage_idx=stage_idx,
                    **stage_config,
                )

        hook_kwargs = dict(input_list=input_list, optim_param=optim_param)
        self.call_hook('after_optimize', **hook_kwargs)
        # collate results
        ret = optim_param

        if return_verts or return_joints or \
                return_full_pose or return_losses:
            eval_ret = self.evaluate(
                input_list=input_list,
                optim_param=optim_param,
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
                           optim_param: dict,
                           epoch_idx: int = -1,
                           stage_idx: int = -1,
                           use_shoulder_hip_only: bool = False,
                           body_weight: float = 1.0,
                           n_iter: int = 1,
                           ftol: float = 1e-4,
                           **kwargs) -> None:
        """Optimize a stage of body model parameters according to
        configuration.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            param_dict (dict):
                A dict of optimizable parameters, whose keys are
                self.__class__.OPTIM_PARAM and values are
                Tensors in batch.
            epoch_idx (int, optional):
                The index of this epoch. For hook and verbose only,
                it will not influence the optimize result.
                Defaults to -1.
            stage_idx (int, optional):
                The index of this stage. For hook and verbose only,
                it will not influence the optimize result.
                Defaults to -1.
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
            n_iter (int, optional):
                Number of iterations. Defaults to 1.
            ftol (float, optional):
                Defaults to 1e-4.

        kwargs:
            Parameter fit flag and stage control keyword arguments.
            Parameter fit args includes fit_{param_key}, while param_key
            is in self.__class__.OPTIM_PARAM. If fit_{param_key} not found
            in kwargs, use default value True.
            Stage control args includes override weights
            of each loss handler.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        """
        stage_config = dict(
            epoch_idx=epoch_idx,
            stage_idx=stage_idx,
            use_shoulder_hip_only=use_shoulder_hip_only,
            body_weight=body_weight,
            n_iter=n_iter,
            ftol=ftol)
        stage_config.update(kwargs)
        hook_kwargs = dict(
            input_list=input_list,
            optim_param=optim_param,
            stage_config=stage_config)
        self.call_hook('before_stage', **hook_kwargs)

        kwargs = kwargs.copy()

        # add individual optimizer choice
        optimizers = {}
        if 'individual_optimizer' not in self.optimizer:
            parameters = OptimizableParameters()
            for key, value in optim_param.items():
                fit_flag = kwargs.pop(f'fit_{key}', True)
                parameters.add_param(key=key, param=value, fit_param=fit_flag)
            optimizers['default_optimizer'] = build_optimizer(
                parameters, self.optimizer)
        else:
            # set an individual optimizer if optimizer config
            # is given and fit_{key} is True
            # update with the default optimizer or ignore otherwise
            # | {key}_opt_config |  fit_{key}  |       optimizer     |
            # | -----------------| ------------| --------------------|
            # |       True       |     True    |   {key}_optimizer   |
            # |       False      |     True    |   default_optimizer |
            # |       True       |     False   |        ignore       |
            # |       False      |     False   |        ignore       |
            self.individual_optimizer = True
            _optim_param = optim_param.copy()
            for key in list(_optim_param.keys()):
                parameters = OptimizableParameters()
                fit_flag = kwargs.pop(f'fit_{key}', False)
                if f'{key}_optimizer' in self.optimizer.keys() and fit_flag:
                    value = _optim_param.pop(key)
                    parameters.add_param(
                        key=key, param=value, fit_param=fit_flag)
                    optimizers[key] = build_optimizer(
                        parameters, self.optimizer[f'{key}_optimizer'])
                    self.logger.info(f'Add an individual optimizer for {key}')
                elif not fit_flag:
                    _optim_param.pop(key)
                else:
                    self.logger.info(f'No optimizer defined for {key}, '
                                     'get the default optimizer')

            if len(_optim_param) > 0:
                parameters = OptimizableParameters()
                if 'default_optimizer' not in self.optimizer:
                    self.logger.error(
                        'Individual optimizer mode is selected but '
                        'some optimizers are not defined. '
                        'Please set the default_optimzier or set optimizer '
                        f'for {_optim_param.keys()}.')
                    raise KeyError
                else:
                    for key in list(_optim_param.keys()):
                        fit_flag = kwargs.pop(f'fit_{key}', True)
                        value = _optim_param.pop(key)
                        if fit_flag:
                            parameters.add_param(
                                key=key, param=value, fit_param=fit_flag)
                    optimizers['default_optimizer'] = build_optimizer(
                        parameters, self.optimizer['default_optimizer'])

        previous_loss = None
        for iter_idx in range(n_iter):
            for optimizer_key, optimizer in optimizers.items():

                def closure():
                    optimizer.zero_grad()

                    betas_video = self.__expand_betas__(
                        batch_size=optim_param['body_pose'].shape[0],
                        betas=optim_param['betas'])
                    expanded_param = {}
                    expanded_param.update(optim_param)
                    expanded_param['betas'] = betas_video
                    loss_dict = self.evaluate(
                        input_list=input_list,
                        optim_param=expanded_param,
                        use_shoulder_hip_only=use_shoulder_hip_only,
                        body_weight=body_weight,
                        **kwargs)

                    if optimizer_key not in loss_dict.keys():
                        self.logger.error(
                            f'Individual optimizer is set for {optimizer_key}'
                            'but there is no loss calculated for this '
                            'optimizer. Please check LOSS_MAPPING and '
                            'make sure respective losses are turned on.')
                        raise KeyError
                    loss = loss_dict[optimizer_key]
                    total_loss = loss_dict['total_loss']

                    loss.backward(retain_graph=True)

                    torch.nn.utils.clip_grad_norm_(
                        parameters=optim_param.values(),
                        max_norm=self.grad_clip)

                    return total_loss

                total_loss = optimizer.step(closure)

            if iter_idx > 0 and previous_loss is not None and ftol > 0:
                loss_rel_change = self.__compute_relative_change__(
                    previous_loss, total_loss.item())
                if loss_rel_change < ftol:
                    if self.verbose:
                        self.logger.info(
                            f'[ftol={ftol}] Early stop at {iter_idx} iter!')
                    break
            previous_loss = total_loss.item()

        stage_config = dict(
            use_shoulder_hip_only=use_shoulder_hip_only,
            body_weight=body_weight,
            n_iter=n_iter,
            ftol=ftol)
        stage_config.update(kwargs)

        betas_video = self.__expand_betas__(
            batch_size=optim_param['body_pose'].shape[0],
            betas=optim_param['betas'])
        expanded_param = optim_param.copy()
        expanded_param['betas'] = betas_video
        loss_dict = self.evaluate(
            input_list=input_list,
            optim_param=expanded_param,
            use_shoulder_hip_only=use_shoulder_hip_only,
            body_weight=body_weight,
            **kwargs)
        hook_kwargs = dict(
            input_list=input_list,
            optim_param=optim_param,
            stage_config=stage_config,
            loss_dict=loss_dict)
        self.call_hook('after_stage', **hook_kwargs)

    def evaluate(self,
                 input_list: List[BaseInput],
                 optim_param: dict,
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
            optim_param (dict):
                A dict of optimizable parameters, whose keys are
                self.__class__.OPTIM_PARAM and values are
                Tensors in batch.
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
        hook_kwargs = dict(input_list=input_list, optim_param=optim_param)
        self.call_hook('before_evaluate', **hook_kwargs)
        ret = {}

        require_verts = self.__check_verts_requirement__(
            input_list, **kwargs) or return_verts
        body_model_kwargs = dict(
            return_verts=require_verts, return_full_pose=return_full_pose)
        body_model_kwargs.update(optim_param)
        body_model_output = self.body_model(**body_model_kwargs)

        model_joints = body_model_output['joints']
        model_joints_convention = self.body_model.keypoint_convention
        model_joints_weights = self.get_keypoint_weight(
            use_shoulder_hip_only=use_shoulder_hip_only,
            body_weight=body_weight,
            **kwargs)
        model_vertices = body_model_output.get('vertices', None)

        loss_dict = self.__compute_loss__(
            input_list=input_list,
            model_joints=model_joints,
            model_joints_convention=model_joints_convention,
            model_joints_weights=model_joints_weights,
            model_vertices=model_vertices,
            reduction_override=reduction_override,
            optim_param=optim_param,
            **kwargs)
        ret.update(loss_dict)

        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        if return_joints:
            ret['joints'] = model_joints

        hook_kwargs = dict(
            input_list=input_list,
            optim_param=optim_param,
            loss_dict=loss_dict)
        self.call_hook('after_evaluate', **hook_kwargs)

        return ret

    def __compute_loss__(self,
                         input_list: List[BaseInput],
                         model_joints: torch.Tensor = None,
                         model_joints_convention: str = '',
                         model_joints_weights: torch.Tensor = None,
                         model_vertices: torch.Tensor = None,
                         optim_param: dict = None,
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
            param_dict (dict):
                A dict of optimizable parameters, whose keys are
                self.__class__.OPTIM_PARAM and values are
                Tensors in batch.
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
        if model_joints_weights is None and \
                model_joints is not None:
            model_joints_weights = torch.ones_like(model_joints[0, :, 0])
        init_handler_input = {
            'model_vertices': model_vertices,
            'model_joints': model_joints,
            'model_joints_convention': model_joints_convention,
            'model_joints_weights': model_joints_weights,
        }
        init_handler_input.update(optim_param)
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
            # e.g. keypoints 3d mse loss
            else:
                for input_inst in input_list:
                    if input_inst.handler_key == handler_key:
                        handler_input['related_input'] = input_inst
                        loss_tensor = handler(**handler_input)
            # if loss computed, record it in losses
            if loss_tensor is not None:
                if loss_tensor.ndim == 3:
                    loss_tensor = loss_tensor.sum(dim=(2, 1))
                elif loss_tensor.ndim == 2:
                    loss_tensor = loss_tensor.sum(dim=-1)
                losses[handler_key] = loss_tensor

        total_loss = 0
        for key, loss in losses.items():
            total_loss = total_loss + loss
        losses['total_loss'] = total_loss

        if self.individual_optimizer:
            losses = self._post_process_loss(losses)
        else:
            losses['default_optimizer'] = total_loss

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

        return losses

    def _post_process_loss(self, losses: dict, **kwargs) -> dict:
        """Process losses and map the losses to respective parameters.

        Args:
            losses (dict): Original loss, use handler_key as keys.

        Returns:
            dict: Processed loss, use parameter names as keys.
                Original keys included.
        """

        for loss_key in list(losses.keys()):
            process_list = LOSS_MAPPING.get(loss_key, [])
            for optimizer_loss in process_list:
                losses[optimizer_loss] = losses[optimizer_loss] + \
                    losses[loss_key] if optimizer_loss in losses \
                    else losses[loss_key]

        losses['default_optimizer'] = losses['total_loss']

        return losses

    def __match_init_batch_size__(self, init_param: torch.Tensor,
                                  default_param: torch.Tensor,
                                  batch_size: int) -> torch.Tensor:
        """A helper function to ensure body model parameters have the same
        batch size as the input keypoints.

        Args:
            init_param:
                Input initial body model parameters, may be None
            default_param:
                Default parameters if init_param is None. Typically
                it is from the body model.
            batch_size:
                Batch size of input.

        Returns:
            param: body model parameters with batch size aligned
        """

        # param takes init values
        param = init_param.detach().clone() \
            if init_param is not None \
            else default_param.detach().clone()

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
                param.shape[1:] != default_param.shape[1:]:
            self.logger.error(f'Shape mismatch: {param.shape} vs' +
                              f' {default_param.shape}')
            raise ValueError
        return param

    def __set_keypoint_indexes__(self) -> None:
        """Set keypoint indexes to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.

        Returns:
            None
        """
        convention = self.body_model.keypoint_convention

        # obtain ignore keypoint indexes
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(
                    keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indexes
        self.face_keypoint_idxs = get_keypoint_idxs_by_part(
            'head', convention=convention)
        left_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'left_hand', convention=convention)
        right_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'right_hand', convention=convention)
        self.hand_keypoint_idxs = [
            *left_hand_keypoint_idxs, *right_hand_keypoint_idxs
        ]
        self.body_keypoint_idxs = get_keypoint_idxs_by_part(
            'body', convention=convention)
        self.shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)
        self.hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [
            *self.shoulder_keypoint_idxs, *self.hip_keypoint_idxs
        ]
        self.foot_keypoint_idxs = get_keypoint_idxs_by_part(
            'foot', convention=convention)

    def get_keypoint_weight(self,
                            use_shoulder_hip_only: bool = False,
                            body_weight: float = 1.0,
                            hand_weight: float = 1.0,
                            face_weight: float = 1.0,
                            shoulder_weight: Union[float, None] = None,
                            hip_weight: Union[float, None] = None,
                            foot_weight: Union[float, None] = None,
                            **kwargs) -> torch.Tensor:
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
            hand_weight (float, optional):
                Weight of hands keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            face_weight (float, optional):
                Weight of face keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            shoulder_weight (float, optional):
                Weight of shoulder keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.
            hip_weight (float, optional):
                Weight of hip keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.
            foot_weight (float, optional):
                Weight of feet keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.

        Returns:
            torch.Tensor: Per keypoint weight tensor of shape (K).
        """
        n_keypoints = self.body_model.get_joint_number()

        # 3rd priority: set body parts weight manually
        # when both body weight and body parts weight set,
        # body parts weight override the body weight
        weight = torch.ones([n_keypoints]).to(self.device)

        # "body" includes "shoulder", "hip" and "foot" keypoints
        weight[self.body_keypoint_idxs] = \
            weight[self.body_keypoint_idxs] * body_weight

        if shoulder_weight is not None:
            weight[self.shoulder_keypoint_idxs] = 1.0
            weight[self.shoulder_keypoint_idxs] = \
                weight[self.shoulder_keypoint_idxs] * shoulder_weight

        if hip_weight is not None:
            weight[self.hip_keypoint_idxs] = 1.0
            weight[self.hip_keypoint_idxs] = \
                weight[self.hip_keypoint_idxs] * hip_weight

        if foot_weight is not None:
            weight[self.foot_keypoint_idxs] = 1.0
            weight[self.foot_keypoint_idxs] = \
                weight[self.foot_keypoint_idxs] * foot_weight

        weight[self.face_keypoint_idxs] = \
            weight[self.face_keypoint_idxs] * face_weight

        weight[self.hand_keypoint_idxs] = \
            weight[self.hand_keypoint_idxs] * hand_weight

        # 2nd priority: use_shoulder_hip_only
        if use_shoulder_hip_only:
            weight = torch.zeros([n_keypoints]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            if shoulder_weight is not None and hip_weight is not None and \
                    body_weight * face_weight * hand_weight == 0.0:
                weight[self.shoulder_keypoint_idxs] = \
                    weight[self.shoulder_keypoint_idxs] * shoulder_weight
                weight[self.hip_keypoint_idxs] = \
                    weight[self.hip_keypoint_idxs] * hip_weight
            else:
                self.logger.error(
                    'use_shoulder_hip_only is deprecated, '
                    'please manually set: body_weight=0.0, face_weight=0.0, '
                    'hand_weight=0.0, shoulder_weight=1.0, hip_weight=1.0 to '
                    'make sure correct weights are set.')
                raise ValueError

        # 1st priority: keypoints ignored
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
        """Check whether vertices are required by loss handlers. For those
        which doesn't need a related input, we take logical or of
        handler.requires_verts(). For those which needs input, we only take
        them when the input is given.

        Args:
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.

        kwargs:
            Stage control keyword arguments, including override weights
            of each loss handler.

        Returns:
            bool: Whether verts are required by any of the handlers.
        """
        any_require_verts = False
        kwargs = kwargs.copy()
        for handler in self.loss_handlers:
            handler_key = handler.handler_key
            loss_weight_override = kwargs.pop(f'{handler_key}_weight', None)
            if self.__skip_handler__(
                    loss_handler=handler,
                    loss_weight_override=loss_weight_override):
                continue
            # e.g. shape prior loss
            if not handler.requires_input():
                any_require_verts = any_require_verts or \
                    handler.requires_verts()
            # e.g. keypoints 3d mse loss
            else:
                for input_inst in input_list:
                    if input_inst.handler_key == handler_key:
                        any_require_verts = any_require_verts or \
                            handler.requires_verts()
        return any_require_verts

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
