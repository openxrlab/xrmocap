import logging
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints


class BaseOptimizer:

    def __init__(self,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class for keypoints3d optimizer.

        Args:
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.verbose = verbose
        self.logger = get_logger(logger)

    def optimize_keypoints3d(self, keypoints: Keypoints) -> Keypoints:
        raise NotImplementedError
