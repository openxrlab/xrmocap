import logging
import torch
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


class RemoveDuplicate(BaseOptimizer):

    def __init__(self,verbose: bool = True,
                 threshold: float = 50,
                 keep: str = 'by_index',
                 logger: Union[None, str, logging.Logger] = None):
        super().__init__(verbose=verbose, logger=logger)
        self.threshold = threshold
        self.keep = keep
        print(">>>Remove duplicate initiated")

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                            **kwargs: dict) -> Keypoints:

        # get kps3d array
        keypoints3d_optim = keypoints3d.clone()
        kps3d = keypoints3d_optim.get_keypoints()
        n_frame, n_max_person, n_kps, _ = kps3d.shape
        

        # calculate simularity between predicted keypoints for each person
        kps3d_optim = np.full((n_frame, n_max_person, n_kps, 4),
                        np.nan)

        for frame_idx in range(n_frame):
            kps3d_frame = kps3d[frame_idx, ...]
            kps3d_frame = kps3d_frame[~np.isnan(kps3d_frame[:,0,0])] # [n_valid_person, 14, 4]
            print("===kps3d frame===: ",kps3d_frame.shape)
            # calculate simularity and remove duplicate
            dist = self.get_kps3d_dist(kps3d_frame) # [n_valid_person, n_valid_person]

            # TODO remove duplicate
            n_valid_person = dist.shape[0]
            remove_idxs = []
            keep_idxs = []
            for person_idx, per_person_dist in enumerate(dist):
                if person_idx not in remove_idxs:
                    print("per person dist: ", per_person_dist)
                    to_remove = np.where(per_person_dist[:] < self.threshold)[0] # index
                    print("person: ", person_idx)
                    print("to remove", to_remove)


                    # which to keep among duplicates?
                    if self.keep == 'by_conf':
                        raise NotImplementedError

                    elif self.keep == 'by_index':
                        # keep lowest index
                        to_keep = np.array([person_idx])
                        to_remove = np.setdiff1d(to_remove,to_keep)
                    
                    remove_idxs.extend(list(to_remove))
                    keep_idxs.extend(list(to_keep))
            
            print(f">>>frame: {frame_idx}; to keep: {keep_idxs}; to remove: {remove_idxs}")

            
            # save to Kps3d
            n_optim_person = len(keep_idxs)
            kps3d_optim[frame_idx, :n_optim_person,
                        ...] = kps3d_frame[keep_idxs, ...]

        keypoints3d_optim.set_keypoints(kps3d_optim)
        keypoints3d_optim.set_mask(kps3d_optim[..., -1] > 0)

        return keypoints3d_optim

    def get_kps3d_dist(self,kps3d: Union[torch.Tensor, np.ndarray], 
        p:int = 2) -> np.ndarray:
        n_valid_person = kps3d.shape[0]
        print("valid person", n_valid_person)
        person = torch.tensor(kps3d[:,0:3].reshape(n_valid_person, -1))
        dist = torch.cdist(person, person) # l2 dist
        print("dist.shape:", dist.shape)

        return dist # [n_valid_person, n_valid_person]
