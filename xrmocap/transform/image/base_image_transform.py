import logging
import torch
from typing import Any, Union
from xrprimer.utils.log_utils import get_logger


class BaseImageTransform(torch.nn.Module):

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class for image transform.

        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__()
        self.logger = get_logger(logger)

    def forward(self, input: Any) -> Any:
        """Forward function of ImageTransform.

        Args:
            input (Any)

        Returns:
            Any
        """
        raise NotImplementedError
