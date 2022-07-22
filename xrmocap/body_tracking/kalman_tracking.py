# yapf: disable
import logging
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.ops.triangulation.point_selection.camera_error_selector import (
    CameraErrorSelector,
)
from xrmocap.ops.triangulation.point_selection.hybrid_kps2d_selector import (
    HybridKps2dSelector,
)
from xrmocap.utils.geometry import compute_iou
from xrmocap.utils.mvpose_utils import get_distance
from .kalman_tracker import KalmanJointTracker

# yapf: enable


class KalmanTracking:

    def __init__(self,
                 state_kps3d: np.ndarray,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialises a tracker using initial body keypoints3d.

        Args:
            state_kps3d (np.ndarray): Initial body keypoints3d,
                in shape (n_person, n_kps2d, 3)
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
            self, n_views: int, kps2d_selector: Union[CameraErrorSelector,
                                                      HybridKps2dSelector],
            kps3d: np.ndarray, dataset, measurement_kps2d: np.ndarray,
            measurement_kps2d_conf: np.ndarray, frame_id: int, n_kps2d: int,
            best_distance_: int, triangulator,
            tracking_bbox: np.ndarray) -> Tuple[np.ndarray, list, list]:
        """Anchor human based matching and return not matched kps2d index.

        Args:
            n_views (int): The number of view.
            kps2d_selector (Union[CameraErrorSelector, HybridKps2dSelector]):
                A camera selector. If it's given, kps2d will be selected
                before triangulation.
            kps3d (np.ndarray): The kps3d on last frame.
            dataset (MemDataset): MemDataset object.
            measurement_kps2d (np.ndarray): The kps2d from 2d detection.
            measurement_kps2d_conf (np.ndarray): The kps2d conf from
                2d detection.
            frame_id (int): Frame id.
            n_kps2d (int): The number of kps2d.
            best_distance_ (int): Maximum matching distance.
            triangulator: AniposelibTriangulator object.
            tracking_bbox (np.ndarray): The bbox of human, and each bbox is
                (x, y, x, y, score).

        Returns:
            multi_kps3d (np.ndarray): Anchor human kps3d.
            not_matched_dimGroup (list): The not matched cumulative number
                of person from different perspectives.
            not_matched_index (list): The kps2d index.
        """
        multi_kps3d = []
        measurement_kps2d[np.isnan(measurement_kps2d)] = 1e-9
        sub_imgid2cam = np.zeros(measurement_kps2d.shape[0], dtype=np.int32)
        dim_group = dataset.dimGroup[frame_id]
        n_cameras = len(dim_group) - 1
        for idx, i in enumerate(range(n_cameras)):
            sub_imgid2cam[dim_group[i]:dim_group[i + 1]] = idx
        bbox_mask = np.ones_like(tracking_bbox[:, 0])
        for i in range(len(sub_imgid2cam)):
            for j in range(i + 1, len(sub_imgid2cam)):
                if sub_imgid2cam[i] == sub_imgid2cam[j]:
                    iou = compute_iou(
                        tracking_bbox[i], tracking_bbox[j], logger=self.logger)
                    if iou > 0.5:
                        bbox_mask[i], bbox_mask[j] = 0, 0
        matched_list = []
        selected_idx = np.where(bbox_mask)[0]

        for human_id in range(kps3d.shape[0]):
            kps2d = triangulator.project(kps3d[human_id])
            sub_matched = []
            for view in range(kps2d.shape[0]):
                tracking_bbox_id = [
                    i for i, x in enumerate(sub_imgid2cam[selected_idx])
                    if x == view
                ]
                best_distance = best_distance_
                best_distance_index = -1
                for idx, measurement_kps2d_ in zip(
                        tracking_bbox_id, measurement_kps2d[tracking_bbox_id]):
                    distance = 0
                    for j in range(n_kps2d):
                        if j not in [1, 2, 3, 4]:
                            distance += get_distance(kps2d[view][j],
                                                     measurement_kps2d_[j])
                        if distance > best_distance:
                            break
                    if distance < best_distance:
                        best_distance = distance
                        best_distance_index = idx
                if best_distance_index != -1:
                    sub_matched.append(best_distance_index)
            if len(sub_matched) > 1.9:
                sub_matched = list(OrderedDict.fromkeys(sub_matched))
                matched_list.append(np.array(sub_matched))
        all_matched_index = [
            i for matched_index in matched_list for i in matched_index
        ]
        not_matched_index = []
        for i in range(measurement_kps2d.shape[0]):
            if i not in all_matched_index:
                not_matched_index.append(i)
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
        for person in matched_list:
            if person.shape[0] < 2:
                continue
            matched_mview_kps2d = np.zeros((n_views, n_kps2d, 2))
            matched_mview_kps2d_mask = np.zeros_like(matched_mview_kps2d[...,
                                                                         0:1])
            matched_mview_kps2d_conf = np.zeros_like(matched_mview_kps2d[...,
                                                                         0:1])

            matched_mview_kps2d[
                sub_imgid2cam[person]] = measurement_kps2d[person]
            matched_mview_kps2d_mask[sub_imgid2cam[person]] = np.ones_like(
                measurement_kps2d[person][..., 0:1])
            matched_mview_kps2d_conf[sub_imgid2cam[
                person]] = measurement_kps2d_conf[sub_imgid2cam[person]]

            selected_mask = kps2d_selector.get_selection_mask(
                np.concatenate((matched_mview_kps2d, matched_mview_kps2d_conf),
                               axis=-1), matched_mview_kps2d_mask)
            kps3d = triangulator.triangulate(matched_mview_kps2d,
                                             selected_mask)
            if not np.isnan(kps3d).all():
                multi_kps3d.append(kps3d)
        multi_kps3d = np.array(multi_kps3d)
        return multi_kps3d, not_matched_dimGroup, not_matched_index
