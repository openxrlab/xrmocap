import logging
from typing import Union
from xrprimer.utils.log_utils import get_logger


class BaseEstimator:
    """Base Estimator."""

    def __init__(self,
                 work_dir: str,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        self.work_dir = work_dir
        self.verbose = verbose
        self.logger = get_logger(logger)

    def run(self) -> None:
        ...
