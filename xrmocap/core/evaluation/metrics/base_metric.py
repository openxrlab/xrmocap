# yapf: disable
import logging
from typing import Union
from xrprimer.utils.log_utils import get_logger

# yapf: enable


class BaseMetric:
    RANK = 0

    def __init__(
        self,
        name: str,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        self.name = name
        self.logger = get_logger(logger)

    def __call__(self, *args, **kwargs):
        return dict()
