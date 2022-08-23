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

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **kwargs: dict) -> Keypoints:
        """Forward function of keypoints3d optimizer.

        Args:
            keypoints3d (Keypoints): Input keypoints3d.
        kwargs:
            Redundant keyword arguments to be
            ignored, including:
                mview_keypoints2d

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        raise NotImplementedError
