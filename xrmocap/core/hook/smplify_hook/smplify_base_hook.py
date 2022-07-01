from mmcv.runner.hooks import Hook
from typing import List, TypeVar

from xrmocap.model.registrant.handler.base_handler import BaseInput

_SMPLify = TypeVar('_SMPLify')


class SMPLifyBaseHook(Hook):
    """Base class of any SMPLify hook."""

    def before_optimize(self, registrant: _SMPLify,
                        input_list: List[BaseInput],
                        optim_param: dict) -> None:
        """When SMPLify is called, this hook will be triggered just before
        optimization starts.

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
        """
        pass

    def after_optimize(self, registrant: _SMPLify, input_list: List[BaseInput],
                       optim_param: dict) -> None:
        """When SMPLify is called, this hook will be triggered just after
        optimization ends.

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
        """
        pass

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
        pass

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
        pass

    def before_evaluate(self, registrant: _SMPLify,
                        input_list: List[BaseInput],
                        optim_param: dict) -> None:
        """This hook will be triggered at the beginning of SMPLify.evaluate().

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
        """
        pass

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
        pass
