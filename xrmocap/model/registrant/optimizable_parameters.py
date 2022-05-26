import torch
from typing import List


class OptimizableParameters():

    def __init__(self):
        """Collects parameters for optimization."""
        self.opt_params = {}

    def add_param(self,
                  key: str,
                  param: torch.Tensor,
                  fit_param: bool = True) -> None:
        """"Set requires_grad, collect the parameter for optimization.

        Args:
            key (str):
                Key of the param.
            param (torch.Tensor):
                Model parameter.
            fit_param (bool):
                Whether to optimize this body model parameter.
                Defaults to True.
        """
        if fit_param:
            param.requires_grad = True
        else:
            param.requires_grad = False
        self.opt_params[key] = param

    def set_param(self, key: str, fit_param: bool = True) -> None:
        """"Set requires_grad of a param in self.opt_params.

        Args:
            key (str):
                Key of the param.
            fit_param (bool):
                Whether to optimize this body model parameter.
                Defaults to True.
        """
        if fit_param:
            self.opt_params[key].requires_grad = True
        else:
            self.opt_params[key].requires_grad = False

    def parameters(self) -> List[torch.Tensor]:
        """Returns all parameters recorded by self. Compatible with mmcv's
        build_parameters()

        Returns:
            List[torch.Tensor]:
                A list of body model parameters for optimization.
        """
        ret_list = []
        for _, value in self.opt_params.items():
            ret_list.append(value)
        return ret_list
