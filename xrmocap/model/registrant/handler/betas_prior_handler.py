import logging
import torch
from typing import TypeVar, Union

from xrmocap.model.loss.builder import build_loss
from .base_handler import BaseHandler

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_BetasPriorLoss = TypeVar('_BetasPriorLoss')


class BetasPriorHandler(BaseHandler):

    def __init__(self,
                 prior_loss: Union[_BetasPriorLoss, dict],
                 handler_key='betas_prior',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a BetasPriorHandler instance compute smpl(x/xd) betas
        parameters, return a loss Tensor.

        Args:
            prior_loss (Union[BetasPriorLoss, dict]):
                An instance of BetasPriorLoss, or a config dict of
                BetasPriorLoss.
            handler_key (str, optional):
                Key of this input-handler pair.
                Defaults to 'betas_prior'.
            device (Union[torch.device, str], optional):
                Device in pytorch. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            TypeError: prior_loss is neither a torch.nn.Module nor a dict.
        """
        super().__init__(handler_key=handler_key, device=device, logger=logger)
        if isinstance(prior_loss, dict):
            self.prior_loss = build_loss(prior_loss)
        elif isinstance(prior_loss, torch.nn.Module):
            self.prior_loss = prior_loss
        else:
            self.logger.error('Type of prior_loss is not correct.\n' +
                              f'Type: {type(prior_loss)}.')
            raise TypeError
        self.prior_loss = self.prior_loss.to(self.device)

    def requires_input(self) -> bool:
        """Whether this handler requires additional input. If not, the
        computation only depends on smpl results.

        Returns:
            bool: Whether this handler requires additional input.
        """
        return False

    def get_loss_weight(self) -> float:
        """Get the weight value of this loss handler.

        Returns:
            float: Weight value.
        """
        loss_weight = self.prior_loss.loss_weight
        return float(loss_weight)

    def __call__(self,
                 betas: torch.Tensor,
                 loss_weight_override: float = None,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs: dict) -> torch.Tensor:
        """Taking smpl(x/xd) parameters, compute loss and return a Tensor.

        Args:
            betas (torch.Tensor):
                The body shape parameters. In shape (batch_size, 10).
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
        betas_prior_loss = self.prior_loss(
            betas=betas,
            loss_weight_override=loss_weight_override,
            reduction_override=reduction_override)
        return betas_prior_loss
