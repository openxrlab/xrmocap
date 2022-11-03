# yapf: disable
import logging
import numpy as np
import torch
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.top_down_association.identity_tracking.builder import (
    BaseTracking, build_identity_tracking,
)
from .base_optimizer import BaseOptimizer

# yapf: enable


class RemoveDuplicate(BaseOptimizer):
    """This optimization tool helps remove duplicate identities by the L2
    distance between 3D keypoints for each person and add tracked identities to
    optimized 3D keypoints."""

    def __init__(
        self,
        verbose: bool = True,
        threshold: float = 2.0,
        keep: str = 'by_index',
        identity_tracking: Union[None, dict, BaseTracking] = None,
        logger: Union[None, str, logging.Logger] = None,
    ):
        """Initialization for RemoveDuplicate optimizor.

        Args:
            verbose (bool, optional):
                Whether to log info. Defaults to True.
            threshold (float, optional):
                Threshold for duplicate 3D keypoints. Defaults to 2.0.
            keep (str, optional):
                Mode to keep 3D keypoints, 'by_index' keeps the lowest index,
                'by_conf' keeps the highest confidence. Defaults to 'by_index'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """

        super().__init__(verbose=verbose, logger=logger)
        self.threshold = threshold
        self.keep = keep

        if self.keep not in {'by_index', 'by_conf'}:
            self.logger.error(f'"{self.keep}" is not a valid mode. '
                              'Please select the mode between '
                              '"by_index" and "by_conf".')
            raise ValueError

        if isinstance(identity_tracking, dict):
            identity_tracking['logger'] = logger
            self.identity_tracking = build_identity_tracking(identity_tracking)
        else:
            self.identity_tracking = identity_tracking

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **kwargs: dict) -> Keypoints:
        """Forward function for RemoveDuplicate optimizor.

        Args:
            keypoints3d (Keypoints):
                Keypoints3d to be optimized.

        Returns:
            Keypoints: Optimized keypoints3d.
        """
        # get kps3d array
        keypoints3d_optim = keypoints3d.clone()
        kps3d = keypoints3d_optim.get_keypoints()
        n_frame, n_max_person, n_kps, _ = kps3d.shape

        kps3d_optim = np.full((n_frame, n_max_person, n_kps, 4), np.nan)

        for frame_idx in range(n_frame):
            kps3d_frame = kps3d[frame_idx, ...]
            kps3d_frame = kps3d_frame[~np.isnan(kps3d_frame[:, 0, 0])]

            # calculate distance and remove duplicate
            dist = self.get_kps3d_dist(kps3d_frame)

            remove_idxs = []
            keep_idxs = []
            for person_idx, per_person_dist in enumerate(dist):
                if person_idx not in remove_idxs \
                        and person_idx not in keep_idxs:
                    to_remove = np.where(
                        per_person_dist[:] < self.threshold)[0]  # index

                    if self.keep == 'by_conf':
                        # keep the highest confidence
                        keep_idx = to_remove[np.argmax(kps3d_frame[to_remove,
                                                                   0, -1])]

                    elif self.keep == 'by_index':
                        # keep the lowest index
                        keep_idx = person_idx

                    to_keep = np.array([keep_idx])
                    to_remove = np.setdiff1d(to_remove, to_keep)

                    remove_idxs.extend(list(to_remove))
                    keep_idxs.extend(list(to_keep))

            # identity tracking
            if self.identity_tracking is not None:
                curr_kps3d = kps3d_frame[keep_idxs, ...]
                frame_identities = self.identity_tracking.query(curr_kps3d)

                # save to Kps3d
                kps3d_optim[frame_idx, frame_identities,
                            ...] = kps3d_frame[keep_idxs, ...]
            else:
                # save to Kps3d
                n_optim_person = len(keep_idxs)
                kps3d_optim[frame_idx, :n_optim_person,
                            ...] = kps3d_frame[keep_idxs, ...]

        keypoints3d_optim.set_keypoints(kps3d_optim)
        keypoints3d_optim.set_mask(kps3d_optim[..., -1] > 0)

        return keypoints3d_optim

    def get_kps3d_dist(self,
                       kps3d: Union[torch.Tensor, np.ndarray],
                       p: int = 2) -> np.ndarray:
        """Calculate the distance between each set of keypoints3d.

        Args:
            kps3d (Union[torch.Tensor, np.ndarray]):
                keypoints3d of the current frame, [n_person, n_kps, 4].
            p (int, optional):
                 p value for the p-norm distance to calculate
                 between each vector pair. Defaults to 2.

        Returns:
            np.ndarray:
                a distance metrix of the size:
                [n_valid_person, n_valid_person].
        """
        n_valid_person = kps3d.shape[0]

        person = torch.tensor(kps3d[:, 0:3].reshape(n_valid_person, -1))
        dist = torch.cdist(person, person, p)  # l2 dist by default

        return dist
