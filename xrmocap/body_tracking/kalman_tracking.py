import logging
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union

from xrmocap.keypoints3d_estimation.lib import pictorial
from xrmocap.utils.log_utils import get_logger
from xrmocap.utils.mvpose_utils import get_distance
from .kalman_tracker import KalmanJointTracker


class KalmanTracking:

    def __init__(self,
                 state_kps3d: np.ndarray,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialises a tracker using initial body keypoints3d.

        Args:
            state_kps3d (np.ndarray): Initial body keypoints3d.
            logger (Union[None, str, logging.Logger], optional):
                Defaults to None.
        """

        self.state_kps3d = state_kps3d
        self.logger = get_logger(logger)
        self.tracker_list = []

        for kps3d in self.state_kps3d:
            self.tracker_list.append(KalmanJointTracker(kps3d))

    def predict(self):
        """Predict next state (prior) using the Kalman filter state propagation
        equations."""
        for tracker in self.tracker_list:
            _ = tracker.predict()

    def update(self, measurement_kps3d: np.ndarray) -> np.ndarray:
        """Updates the state vector with measurement body keypoints3d.

        Args:
            measurement_kps3d (np.ndarray): measurement body keypoints3d.

        Returns:
            np.ndarray: keypoints3d after kalman filter.
        """
        new_kps3d_list = []

        for tracker, measurement_kps3d_ in zip(self.tracker_list,
                                               measurement_kps3d):
            tracker.update(measurement_kps3d_)
            kps3d_after_kalman = tracker.get_update()
            new_kps3d_list.append(kps3d_after_kalman[:, :, 0])

        return np.array(new_kps3d_list)

    def get_measurement_kps3d(
            self, kps3d: np.ndarray, dataset, measurement_kps2d: np.ndarray,
            frame_id: int, n_kps2d: int, convention: str, best_distance_: int,
            triangulator) -> Tuple[np.ndarray, list, np.ndarray, list]:
        """Anchor human based matching and return new keypoints 2d.

        Args:
            kps3d (np.ndarray): The kps3d on last frame.
            dataset: MemDataset object.
            measurement_kps2d (np.ndarray): The kps2d from 2d detection.
            frame_id (int): Frame id.
            n_kps2d (int): The number of kps2d.
            convention (str): Convention name of the keypoints,
                can be found in KEYPOINTS_FACTORY.
            best_distance_ (int): Maximum matching distance.
            triangulator: AniposelibTriangulator object.

        Returns:
            multi_kps3d (np.ndarray): Anchor human kps3d.
            not_matched_dimGroup (list): The not matched cumulative number
                of person from different perspectives.
            not_matched_kps2d (np.ndarray): The kps2d that are not matched.
            not_matched_index (list): The kps2d index.
        """
        kps3d[np.isnan(kps3d)] = 1e-9
        measurement_kps2d[np.isnan(measurement_kps2d)] = 1e-9
        sub_imgid2cam = np.zeros(measurement_kps2d.shape[0], dtype=np.int32)
        dim_group = dataset.dimGroup[frame_id]
        n_cameras = len(dim_group) - 1
        for idx, i in enumerate(range(n_cameras)):
            sub_imgid2cam[dim_group[i]:dim_group[i + 1]] = idx
        matched_list = []

        for human_id in range(kps3d.shape[0]):
            kps2d = triangulator.project(kps3d[human_id])
            sub_matched = []
            for view in range(kps2d.shape[0]):
                best_distance = best_distance_
                best_distance_index = -1
                for i, measurement_kps2d_ in enumerate(measurement_kps2d):
                    distance = 0
                    for j in range(n_kps2d):
                        if j not in [1, 2, 3, 4]:
                            distance += get_distance(kps2d[view][j],
                                                     measurement_kps2d_[j])
                        if distance > best_distance:
                            break
                    if distance < best_distance:
                        best_distance = distance
                        best_distance_index = i
                if best_distance_index != -1:
                    sub_matched.append(best_distance_index)
            if len(sub_matched) > 1.9:
                sub_matched = list(OrderedDict.fromkeys(sub_matched))
                matched_list.append(np.array(sub_matched))
        all_matched_index = [
            i for matched_index in matched_list for i in matched_index
        ]
        not_matched_index = []
        not_matched_kps2d = []
        for i, measurement_kps2d_ in enumerate(measurement_kps2d):
            if i not in all_matched_index:
                not_matched_index.append(i)
                not_matched_kps2d.append(measurement_kps2d_)
        not_matched_sub_imgid2cam = sub_imgid2cam[not_matched_index]
        not_matched_dimGroup = [0 for _ in range(n_cameras)]
        for i in not_matched_sub_imgid2cam:
            not_matched_dimGroup[i] += 1
        not_matched_dimGroup.insert(0, 0)
        for i, _ in enumerate(not_matched_dimGroup):
            if i == 0:
                continue
            not_matched_dimGroup[i] = not_matched_dimGroup[i] + \
                not_matched_dimGroup[i-1]
        multi_kps3d, _ = pictorial.hybrid_kernel(
            dataset,
            matched_list,
            measurement_kps2d,
            sub_imgid2cam,
            frame_id,
            keypoint_num=n_kps2d,
            convention=convention)
        multi_kps3d = [multi_kps3d[i].T for i in range(len(multi_kps3d))]
        multi_kps3d = np.array(multi_kps3d)
        return multi_kps3d, not_matched_dimGroup, np.array(
            not_matched_kps2d), not_matched_index
