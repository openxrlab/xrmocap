import logging
from typing import Union

import numpy as np

from xrmocap.utils.log_utils import get_logger


class BaseSelector():

    def __init__(self,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        self.verbose = verbose
        self.logger = get_logger(logger)

    def get_selection_mask(
            self,
            points: Union[np.ndarray, list, tuple],
            logger: Union[None, str, logging.Logger] = None) -> np.ndarray:
        raise NotImplementedError
