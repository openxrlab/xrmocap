import logging
from typing import List, Union
from xrprimer.data_structure import Keypoints
from xrprimer.utils.log_utils import get_logger


class BaseTracking:

    def __init__(self,
                 verbose: bool = False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class for 3D identity tracking.

        Args:
            verbose (bool, optional):
                Whether to print individual losses during registration.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.verbose = verbose
        self.logger = get_logger(logger)

    def query(self, association_list: List[List[int]], keypoints3d: Keypoints,
              **kwargs: dict) -> List[int]:
        """Query identities, pass information about multi-person multi-view
        association as input, get a list of indentities.

        Args:
            association_list (List[List[int]]):
                A nested list of association result,
                in shape [n_person, n_view], and
                association_list[i][j] = k means
                the k-th 2D perception in view j
                is a 2D obersevation of person i.
            keypoints3d (List[Keypoints]):
                An instance of class Keypoints3d,
                whose n_person == len(association_list)
                and n_frame == 1.
        kwargs:
            Keyword args to be ignored.

        Returns:
            List[int]:
                A list of indentities, whose length
                is equal to len(association_list).
        """
        raise NotImplementedError
