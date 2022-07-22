import logging
import torch
from typing import Union
from xrprimer.utils.log_utils import get_logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class BaseInput:

    def __init__(self, handler_key='base_handler') -> None:
        """Construct an input instance for BaseHandler.

        Args:
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'base_handler'.
        """
        self.handler_key = handler_key

    def get_batch_size(self) -> int:
        """Get batch size of the input.

        Raises:
            NotImplementedError:
                BaseInput has not been implemented.

        Returns:
            int: batch_size
        """
        raise NotImplementedError


class BaseHandler:

    def __init__(self,
                 handler_key='base_handler',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a BaseHandler instance compute smpl(x/xd) parameters and
        BaseInput, return a loss Tensor.

        Args:
            handler_key (str, optional):
                Key of this input-handler pair. This input will
                be assigned to a handler who has the same key.
                Defaults to 'base_handler'.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.handler_key = handler_key
        self.device = device
        self.logger = get_logger(logger)

    def requires_input(self) -> bool:
        """Whether this handler requires additional input. If not, the
        computation only depends on smpl results.

        Returns:
            bool: Whether this handler requires additional input.
        """
        return True

    def requires_verts(self) -> bool:
        """Whether this handler requires body_model vertices.

        Returns:
            bool: Whether this handler requires body_model vertices.
        """
        return False

    def get_loss_weight(self) -> float:
        """Get the weight value of this loss handler.

        Returns:
            float: Weight value.
        """
        return 0.0

    def __call__(self,
                 related_input: BaseInput,
                 loss_weight_override: float = None,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs: dict) -> torch.Tensor:
        """Taking BaseInput and smpl(x/xd) parameters, compute loss and return
        a Tensor.

        Args:
            related_input (BaseInput):
                An instance of BaseInput, having the same
                key as self.
            loss_weight_override (float, optional):
                Override the global weight of this loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional):
                Override the reduction method of this loss.
                Defaults to None.

        kwargs:
            Redundant smpl(x/d) keyword arguments to be
            ignored, including:
            model_joints, model_joints_convention.
            We suggest claiming required arguments explicitly,
            not to get arg from kwargs.

        Raises:
            NotImplementedError:
                BaseHandler has not been implemented.

        Returns:
            torch.Tensor:
                A Tensor of loss result.
        """
        raise NotImplementedError
