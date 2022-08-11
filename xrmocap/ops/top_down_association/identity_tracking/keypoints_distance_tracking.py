import logging
import numpy as np
from typing import List, Union

from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx
from .base_tracking import BaseTracking


class KeypointsDistanceTracking(BaseTracking):

    def __init__(self,
                 tracking_distance: np.float,
                 tracking_kps3d_convention: str,
                 tracking_kps3d_name: List[str],
                 verbose: bool = False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """3D identity tracking based on distance between keypoints. This
        method assigns identities of the closest person in last frame.

        Args:
            tracking_distance (np.float):
                The distance considered to be the same kps3d.
            tracking_kps3d_convention (str): The convention of kps3d.
            tracking_kps3d_name (List[str]): The name of the tracked kps3d.
            verbose (bool, optional):
                Whether to print individual losses during registration.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseTracking.__init__(self, verbose=verbose, logger=logger)
        self.tracking_distance = tracking_distance
        self.tracking_kps_idx = []
        for name in tracking_kps3d_name:
            self.tracking_kps_idx.append(
                get_keypoint_idx(
                    name=name, convention=tracking_kps3d_convention))
        self.tracking_kps3d = None

    def query(self, curr_kps3d, **kwargs: dict) -> List[int]:
        ret_list = []
        if self.tracking_kps3d is None:
            self.tracking_kps3d = curr_kps3d
            ret_list = [i for i in range(curr_kps3d.shape[0])]
            return ret_list
        max_identity = self.tracking_kps3d.shape[0]
        dist_mat = [[] for _ in range(max_identity)]
        for person_t_idx, person_t_kps3d in enumerate(self.tracking_kps3d):
            for person_c_kps3d in curr_kps3d:
                dist_mat[person_t_idx].append(
                    np.linalg.norm(
                        person_t_kps3d[self.tracking_kps_idx] -
                        person_c_kps3d[self.tracking_kps_idx],
                        ord=2))
        for person_c_idx in range(len(dist_mat[0])):
            dist_list = []
            for person_t_idx in range(len(dist_mat)):
                dist_list.append(dist_mat[person_t_idx][person_c_idx])
            min_dist = min(dist_list)

            if min_dist < self.tracking_distance:
                index = dist_list.index(min_dist)
                ret_list.append(index)
                self.tracking_kps3d[index] = curr_kps3d[person_c_idx].copy()
            else:
                max_identity += 1
                self.tracking_kps3d = np.concatenate(
                    (self.tracking_kps3d,
                     curr_kps3d[person_c_idx][np.newaxis]),
                    axis=0)
                ret_list.append(max_identity - 1)
        return ret_list
