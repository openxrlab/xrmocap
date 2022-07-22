import logging
import prettytable
from typing import List, TypeVar, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.model.registrant.handler.base_handler import BaseInput
from .smplify_base_hook import SMPLifyBaseHook

_SMPLify = TypeVar('_SMPLify')


class SMPLifyVerboseHook(SMPLifyBaseHook):
    """A hook for logging SMPLify information, controlled by logger level and
    SMPLify attributes."""

    def before_stage(self, registrant: _SMPLify, input_list: List[BaseInput],
                     optim_param: dict, stage_config: dict) -> None:
        """This hook will be triggered at the beginning of
        SMPLify.__optimize_stage__().

        Args:
            registrant (_SMPLify):
                The Registrant instance.
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            optim_param (dict):
                A dictionary that includes body model parameters,
                and optional attributes such as vertices and joints.
                Its keys can be found in
                registrant.__class__.OPTIM_PARAM.
            stage_config (dict):
                A dictionary that includes stage configurations.
        """
        if registrant.verbose:
            epoch_idx = stage_config['epoch_idx']
            stage_idx = stage_config['stage_idx']
            registrant.logger.info(f'epoch {epoch_idx}, stage {stage_idx}')

    def after_stage(self, registrant: _SMPLify, input_list: List[BaseInput],
                    optim_param: dict, stage_config: dict,
                    loss_dict: dict) -> None:
        """This hook will be triggered at the end of
        SMPLify.__optimize_stage__().

        Args:
            registrant (_SMPLify):
                The Registrant instance.
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            optim_param (dict):
                A dictionary that includes body model parameters,
                and optional attributes such as vertices and joints.
                Its keys can be found in
                registrant.__class__.OPTIM_PARAM.
            stage_config (dict):
                A dictionary that includes stage configurations.
            loss_dict (dict):
                A dict that contains all losses of the last evaluation.
        """
        if registrant.verbose:
            SMPLifyVerboseHook.log_losses(loss_dict, registrant.logger)

    def after_evaluate(self, registrant: _SMPLify, input_list: List[BaseInput],
                       optim_param: dict, loss_dict: dict) -> None:
        """This hook will be triggered at the end of SMPLify.evaluate().

        Args:
            registrant (_SMPLify):
                The Registrant instance.
            input_list (List[BaseInput]):
                Additional input for loss handlers. Each element is
                an instance of subclass of BaseInput.
            optim_param (dict):
                A dictionary that includes body model parameters,
                and optional attributes such as vertices and joints.
                Its keys can be found in
                registrant.__class__.OPTIM_PARAM.
            loss_dict (dict):
                A dict that contains all losses of this evaluation.
        """
        if registrant.verbose and registrant.info_level == 'step':
            SMPLifyVerboseHook.log_losses(loss_dict, registrant.logger)

    @staticmethod
    def log_losses(loss_dict: dict,
                   logger: Union[None, str, logging.Logger] = None) -> None:
        """Log the losses in loss_dict.

        Args:
            loss_dict (dict):
                A dict that contains all losses of this evaluation.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        logger = get_logger(logger)
        table = prettytable.PrettyTable()
        table.field_names = ['Loss name', 'Loss value']
        for key, value in loss_dict.items():
            if isinstance(value, float) or \
                    isinstance(value, int) or \
                    len(value.shape) == 0:
                table.add_row([key, f'{value:.6f}'])
            else:
                table.add_row([key, 'Not a scalar'])
        table = '\n' + table.get_string()
        logger.info(table)
