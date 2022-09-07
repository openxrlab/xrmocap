import logging
import numpy as np
from typing import Tuple, Union
from xrprimer.utils.log_utils import get_logger


class BaseMatching:

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class for association matching.

        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)

    def __call__(self, mview_kps2d: np.ndarray) -> Tuple[list, list, list]:
        """Compute method of Matching instance. Giving multi-view kps2d, it
        will return the match results.

        Args:
            mview_kps2d (np.ndarray):
                mview_kps2d (np.ndarray): Multi-view keypoints 2d array,
                in shape [n_view, n_person, n_kps2d, 2].

        Raises:
            NotImplementedError:
                BaseMatching has not been implemented.

        Returns:
            Tuple[list, list, list]:
                matched_kps2d (list): Matched human keypoints.
                matched_human_idx (list): Matched human index.
                matched_observation (list): Matched human observed
                    camera number.
        """
        raise NotImplementedError
