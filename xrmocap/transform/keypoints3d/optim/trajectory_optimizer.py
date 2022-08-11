import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


class TrajectoryOptimizer(BaseOptimizer):

    def __init__(self,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Look for kps3d that deviate from the trajectory, and replace it by
        interpolation.

        Args:
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)

    def optimize_keypoints3d(self, keypoints: Keypoints) -> Keypoints:
        if keypoints.dtype == 'numpy':
            keypoints_np = keypoints
        else:
            keypoints_np = keypoints.to_numpy()
            self.logger.warning(
                'TrajectoryOptimizer only support numpy kps for now,' +
                ' the input kps has been converted to numpy.')
        ret_keypoints = keypoints_np.clone()
        ret_kps3d_arr = ret_keypoints.get_keypoints()
        n_person = keypoints_np.get_person_number()
        for person_idx in range(n_person):
            kps3d_arr = keypoints_np.get_keypoints()[:, person_idx]
            optimized_kps3d = self.check_kps3d(kps3d_arr)
            ret_kps3d_arr[:, person_idx] = optimized_kps3d
        ret_keypoints.set_keypoints(ret_kps3d_arr)
        return ret_keypoints

    def check_kps3d(self, kps3d_arr: np.ndarray) -> np.ndarray:
        kps3d = kps3d_arr[..., :3]
        kps3d_score = kps3d_arr[..., 3:4]
        n_frame = kps3d_arr.shape[0]
        n_kps3d = kps3d_arr.shape[1]
        person_nan = list(np.sum(kps3d_score[..., 0], axis=1) == 0)
        kps3d[person_nan] = np.nan

        for frame_idx in range(2, n_frame - 1):
            for kps3d_idx in range(n_kps3d):
                if np.isnan(kps3d[frame_idx, kps3d_idx]).all():
                    continue
                calc_curr_dist = True
                for i in range(1, 7):
                    if frame_idx - i < 1:
                        break
                    curr_dist = np.linalg.norm(
                        kps3d[frame_idx, kps3d_idx] -
                        kps3d[frame_idx - i, kps3d_idx],
                        ord=2) / i
                    if curr_dist > 0:
                        for j in range(frame_idx - i - 1, frame_idx - i - 7,
                                       -1):
                            if not np.isnan(kps3d[j, kps3d_idx]).all():
                                dist_threshold = 2 * np.linalg.norm(
                                    kps3d[frame_idx - i, kps3d_idx] -
                                    kps3d[j, kps3d_idx],
                                    ord=2) / (
                                        frame_idx - i - j)
                                if curr_dist > dist_threshold:
                                    kps3d[frame_idx, kps3d_idx] = np.nan
                                calc_curr_dist = False
                                break
                    if not calc_curr_dist:
                        break
        kps3d_score[person_nan] = np.nan
        return np.concatenate((kps3d, kps3d_score), axis=-1)
