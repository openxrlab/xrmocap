import logging
import torch
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


class RemoveDuplicate(BaseOptimizer):

    def __init__(self,verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None):
        super().__init__(verbose=verbose, logger=logger)

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                            **kwargs: dict) -> Keypoints:

        # convert keypoints to numpy
        kps3d = keypoints3d.get_keypoints()
        n_frame, n_max_person, n_kps, _ = kps3d.shape
        

        # calculate simularity between predicted keypoints for each person
        kps3d_optim = np.full((n_frame, self.n_max_person, self.n_kps, 4),
                        np.nan)

        for frame_idx in range(n_frame):
            kps3d_frame_optim = []
            kps3d_frame = kps3d[frame_idx]
            kps3d_frame = kps3d_frame[~np.isnan(kps3d_frame)] # [n_valid_person, 14, 4]

            # calculate simularity and remove duplicate
            dist = self.get_kps3d_dist(kps3d_frame) # [n_valid_person, n_valid_person]
            # TODO remove duplicate

            # save to Kps3d
            n_optim_person = kps3d_frame_optim.shape[0]
            kps3d_optim[frame_idx, :n_optim_person,
                        ...] = kps3d_frame_optim

        return kps3d_optim

    def get_kps3d_dist(self,kps3d):
        n_valid_person = kps3d_frame.shape[0]
        person = kps3d.reshape(n_valid_person, -1)
        dist = torch.cdist(person, person) # [n_valid_person, n_valid_person]
