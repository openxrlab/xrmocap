import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_optimizer import BaseOptimizer


class TrajectoryOptimizer(BaseOptimizer):

    def __init__(self,
                 n_max_frame: int = 9,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Look for kps3d that deviate from the trajectory, and replace it by
        interpolation.

        Args:
            n_max_frame (int, optional):
                Find the maximum range of valid points. Defaults to 9.
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        self.n_max_frame = n_max_frame

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **kwargs: dict) -> Keypoints:
        """Forward function of keypoints3d optimizer.

        Args:
            keypoints3d (Keypoints): Input keypoints3d.
        kwargs:
            Redundant keyword arguments to be
            ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        if keypoints3d.dtype == 'numpy':
            keypoints3d_np = keypoints3d
        else:
            keypoints3d_np = keypoints3d.to_numpy()
            self.logger.warning(
                'TrajectoryOptimizer only support numpy kps for now,' +
                ' the input kps has been converted to numpy.')
        ret_keypoints3d = keypoints3d_np.clone()
        ret_kps3d_arr = ret_keypoints3d.get_keypoints()
        n_person = keypoints3d_np.get_person_number()
        for person_idx in range(n_person):
            kps3d_arr = keypoints3d_np.get_keypoints()[:, person_idx]
            kps3d_mask = keypoints3d_np.get_mask()[:, person_idx]
            optimized_kps3d = self.check_kps3d(kps3d_arr, kps3d_mask)
            ret_kps3d_arr[:, person_idx] = optimized_kps3d
        ret_keypoints3d.set_keypoints(ret_kps3d_arr)
        return ret_keypoints3d

    def check_kps3d(self, kps3d_arr: np.ndarray,
                    kps3d_mask: np.ndarray) -> np.ndarray:
        kps3d = kps3d_arr[..., :3]
        kps3d_score = kps3d_arr[..., 3:4]
        n_frame = kps3d_arr.shape[0]
        n_kps3d = kps3d_arr.shape[1]
        person_nan = np.where(np.sum(kps3d_mask, axis=1) == 0)[0]
        kps3d[person_nan] = np.nan

        for frame_idx in range(2, n_frame - 1):
            for kps3d_idx in range(n_kps3d):
                if np.isnan(kps3d[frame_idx, kps3d_idx]).all():
                    continue
                calc_curr_dist = True
                for i in range(1, self.n_max_frame):
                    if frame_idx - i < 1:
                        break
                    curr_dist = np.linalg.norm(
                        kps3d[frame_idx, kps3d_idx] -
                        kps3d[frame_idx - i, kps3d_idx],
                        ord=2) / i
                    if curr_dist > 0:
                        for j in range(frame_idx - i - 1,
                                       frame_idx - i - self.n_max_frame, -1):
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
