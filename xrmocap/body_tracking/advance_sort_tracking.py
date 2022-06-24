# yapf:disable
import logging
import numpy as np
from typing import List, Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.limbs import get_limbs_from_keypoints
from xrmocap.utils.log_utils import get_logger
from .kalman_tracker import KalmanJointTracker
from .matching.builder import build_matching

# yapf:enable


def out_of_field(point: np.ndarray) -> bool:
    """Use prior knowledge to decide whether the H projected 2d point is inside
    field, dimension in cm.

    Args:
        point (np.ndarray): 2d point

    Returns:
        bool: whether the point is inside field
    """
    if 0 < point[0] < 1500 and 0 < point[1] < 1400:
        return False
    else:
        return True


class Human:

    def __init__(self, frame_idx: int, kps2d: np.ndarray, kps3d: np.ndarray,
                 track_id: np.ndarray, convention: str) -> None:
        """Human class, including all the functions related to human object.

        Args:
            frame_idx (int): Frame id.
            kps2d (np.ndarray): 2d keypoints, in shape (n_view, n_kps2d, 2).
            kps3d (np.ndarray): 3d keypoints.
            track_id (np.ndarray): Track id.
            convention (str): kps2d factory.
        """
        # maximum age to be moved to permanent history
        self.max_age = 3
        # minimum hits to be considered as confirmed
        self.min_hits = 3
        # frame number for bone constrained triangulation
        self.optim_frame = 25

        self.kps3d_history = [kps3d]
        self.kps2d_history = [kps2d]
        self.kps3d_optim_history = [kps3d]
        self.track_id_history = [track_id]
        self.track_id_history_ = [track_id]
        self.gt_track_id = track_id[~np.isnan(track_id)][0]
        self.start_frame = frame_idx
        self.end_frame = -1
        self.tracker = KalmanJointTracker(kps3d)

        conn_length = {
            'left_lower_leg': 0.40289658,
            'left_thigh': 0.40554457,
            'right_lower_leg': 0.41222511,
            'right_thigh': 0.40550518,
            'left_upperarm': 0.43427501,
            'right_upperarm': 0.33949204,
            'left_forearm': 0.29616352,
            'right_forearm': 0.36330972
        }

        kps2d_score = np.ones((kps2d.shape[0], kps2d.shape[1], 1))
        kps2d = np.concatenate((kps2d, kps2d_score), axis=2)
        keypoints2d = Keypoints(
            kps=np.expand_dims(kps2d, axis=0), convention=convention)

        if convention == 'coco':
            limbs = get_limbs_from_keypoints(
                keypoints=keypoints2d, fill_limb_names=True)
            conn = limbs.get_connections()
            all_conn_dict = limbs.get_connections_by_names()
            selected_conn = []
            for key, value in all_conn_dict.items():
                if key in conn_length.keys():
                    selected_conn.append(value)
            self.conn = [(u, v) for u, v in conn]
            self.conn_idx = np.transpose(np.array(selected_conn))
        else:
            raise ValueError

        self.standard_length = np.array([v for _, v in conn_length.items()])

    # error definition
    def error_track(self, new_track_id):
        error = new_track_id == self.track_id_history[-1]
        error = np.array(error).astype(int)
        return -error.sum()

    @staticmethod
    def error_welsch(n_views, total_view):
        dom = (total_view - 1) / 2
        return np.exp(-0.5 * (n_views / dom)**2) - 0.4

    def error_key_p3(self, new_key_p3):
        pred_key_p3 = self.tracker.history[-1]
        return np.linalg.norm(new_key_p3 - pred_key_p3, axis=1).mean()

    # kalman filter operation and status
    def kalman_predict(self):
        return self.tracker.predict()

    def kalman_update(self, kps3d):
        return self.tracker.update(kps3d)

    def is_confirmed(self):
        return self.tracker.hits >= self.min_hits

    def should_remove(self):
        return self.tracker.time_since_update >= self.max_age

    def archive(self, frame_id):
        self.end_frame = frame_id
        assert (len(self.kps3d_history) == self.end_frame - self.start_frame)

    # main update function
    def update(self, new_key_p2, new_key_p3, new_track_id):
        self.kps2d_history.append(new_key_p2)
        self.kps3d_history.append(new_key_p3)
        self.track_id_history.append(new_track_id)
        self.track_id_history_.append(new_track_id)

    # Other utilities
    def get_optim_key(self, camera_group):  # not available
        """TODO:Implementing temporal optimization.

        Args:
            camera_group (_type_): _description_

        Returns:
            _type_: temporal optimized skeleton
        """
        input_kps2d = np.transpose(
            np.stack(self.kps2d_history[-self.optim_frame:]),
            (1, 0, 2, 3))  # (C, N, J, 2)
        optim_kps3d = camera_group.triangulate_optim(
            input_kps2d, constraints=self.conn, scale_smooth=4)
        return optim_kps3d[-1]

    def get_homo_threshold(self):
        """Vary homography threshold based on feet height.

        Returns:
            anchor, search radius
        """
        xy_coords = self.tracker.history[-1][27:29].mean(axis=0)[:2]
        min_z = self.tracker.history[-1][:, 2].min()
        cam_height = 2
        proj_length = 1200
        search_radius = (proj_length / cam_height) * min_z + 100
        return xy_coords * 100, search_radius

    def is_ratio_human(self) -> bool:
        """Check the reconsturcted bone length vs standard bone length, ratio
        regards to length of left thigh bone.

        Returns:
            bool: whether the ratio shows the skeleton is human.
        """
        kps3d = self.kps3d_history[-1]
        bone_segment = kps3d[self.conn_idx]
        length = np.linalg.norm(bone_segment[0] - bone_segment[1], axis=1)
        error_human_bone = abs(length / length[0] - self.standard_length /
                               self.standard_length[0]).mean()
        return error_human_bone < 0.1

    def is_human(self, always_true=False, human_bone_thres=0.5) -> bool:
        """Check the reconsturcted bone length vs standard bone length, using
        absolute difference.

        Args:
            always_true (bool, optional): Defaults to False.
            human_bone_thres (float, optional): Defaults to 0.5.

        Returns:
            bool: whether the difference shows the skeleton is human.
        """
        if always_true:
            return True
        kps3d = self.kps3d_history[-1]
        bone_segment = kps3d[self.conn_idx]
        length = np.linalg.norm(bone_segment[0] - bone_segment[1], axis=1)
        error_human_bone = abs(length - self.standard_length).mean()

        return error_human_bone < human_bone_thres

    def within_field(self) -> bool:
        """Check whether the reconstructed skeleton is within the field,
        dimension in meters.

        Returns:
            bool
        """
        xy_coords = self.kps3d_history[-1][27:29].mean()[:2]
        if 0 < xy_coords[0] < 15 and 0 < xy_coords[1] < 14:
            return True
        else:
            return False

    def check_track(self):
        track = self.track_id_history[-1]
        average = track.mean()
        return average == track[0]


class AdvanceSort:

    def __init__(self,
                 cfg,
                 n_kps2d: int,
                 dataset,
                 triangulator,
                 start_frame=0,
                 use_homo=False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Sets key parameters for AdvanceSort. As implemented in
        https://github.com/abewley/sort but with some modifications.

        Args:
            n_kps2d (int): The number of keypoints
            dataset: MemDataset.
            triangulator (AniposelibTriangulator): AniposelibTriangulator.
            start_frame (int, optional): Defaults to 0.
            use_homo (bool, optional): use homography related operation
                for reduced graph. Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.cfg = cfg
        self.dataset = dataset
        self.frame_count = start_frame
        self.n_kps2d = n_kps2d
        self.use_homo = use_homo
        self.triangulator = triangulator
        self.logger = get_logger(logger)
        self.always_is_human = 0
        self.dataset_name = cfg.data.name
        self.convention = cfg.kps2d_convention
        self.verbose = cfg.advance_sort_tracking.verbose
        self.match_2d_2d_thres = cfg.advance_sort_tracking[
            self.dataset_name].match_2d_2d_thresholds
        self.match_2d_3d_thres = cfg.advance_sort_tracking[
            self.dataset_name].match_2d_3d_thresholds
        self.human_bone_thresholds = cfg.advance_sort_tracking[
            self.dataset_name].human_bone_thresholds
        self.gt_label = 1 if not cfg.advance_sort_tracking.eval_on_acad else 0
        self.welsch_weight = cfg.advance_sort_tracking[
            self.dataset_name].welsch_weights
        self.kps3d_weight = cfg.advance_sort_tracking[
            self.dataset_name].kps3d_weights
        self.track_weight = cfg.advance_sort_tracking[
            self.dataset_name].track_weights

        self.human = []
        self.matched_list = []
        self.record_error = []
        self.temporary_history = []
        self.permanent_history = []

    def update(self):
        """Requires: this method must be called once for each frame even with
            empty detections.
           Note:as in practical realtime, the detector doesn't run on every
            single frame
        """
        confirmed_human = []
        unconfirmed_human = []
        current_human = []
        self.matched_list.append([])
        self.record_error.append([])

        # Kalman filter predict
        for human in self.human:
            human.kalman_predict()
            if human.is_confirmed():
                confirmed_human.append(human)
            else:
                unconfirmed_human.append(human)

        # First temporal matching
        data = self.dataset.get_tracking_data(self.frame_count, self.n_kps2d)
        matched_list1, matched1, matched_dets, unmatched1, error1 = \
            self.associate_detections_to_human(confirmed_human,
                                               data,
                                               self.triangulator,
                                               use_homo=self.use_homo,
                                               logger=self.logger)
        self.matched_list[-1].extend(matched_list1)
        self.record_error[-1].extend(error1)
        current_human.extend(matched1)
        for human in unmatched1:
            if human.should_remove():
                human.end_frame = self.frame_count + 1
                self.permanent_history.append(human)
            else:
                current_human.append(human)
        # Second temporal matching
        matched_list2, matched2, matched_dets2, unmatched2, error2 = \
            self.associate_detections_to_human(unconfirmed_human,
                                               data,
                                               self.triangulator,
                                               matched_dets,
                                               self.use_homo,
                                               logger=self.logger)
        self.matched_list[-1].extend(matched_list2)
        self.record_error[-1].extend(error2)
        current_human.extend(matched2)
        # Create and initialise new human for unmatched detections
        matched_list3, new_human, error3 = self.generate_new_human(
            self.n_kps2d,
            data,
            self.triangulator,
            matched_dets2,
            self.use_homo,
            self.convention,
            logger=self.logger)
        self.matched_list[-1].extend(matched_list3)
        self.record_error[-1].extend(error3)
        current_human.extend(new_human)
        self.human = current_human
        self.frame_count += 1

    def generate_new_human(
        self,
        n_kps2d: int,
        data: tuple,
        triangulator,
        detection_mask: Union[None, np.ndarray] = None,
        use_homo: bool = True,
        convention: str = 'coco',
        logger: Union[None, str, logging.Logger] = None
    ) -> Tuple[list, List[Human], list]:
        """Implementation for 2d-2d association.

        Args:
            n_kps2d (int): The number of keypoints
            data (tuple): The input data
            triangulator (AniposelibTriangulator): AniposelibTriangulator
            detection_mask (Union[None, np.ndarray], optional): The 2d detected
                mask indicating unmatched observations. Defaults to None.
            use_homo (bool, optional): Whether to use homography related
                operaation for reduced graph. Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                    Logger for logging. If None, root logger will be selected.
                    Defaults to None.

        Returns:
            Tuple[list, List[Human], list]:
                matched_list: Matched person index.
                matched: Human object.
                record_error: 2d-2d matching min error.
        """
        logger = get_logger(logger)
        homo_points, mview_kps2d, track_id, frame_id = data
        n_cameras, n_human, _, _ = mview_kps2d.shape
        # Instantiate the matrix
        if detection_mask is not None:
            mark_mat = detection_mask
        else:
            mark_mat = np.zeros_like(mview_kps2d[:, :, 0, 0])
            # Filter homo projection based on field dimension
            if use_homo:
                for i in range(n_cameras):
                    for j in range(n_human):
                        if out_of_field(homo_points[i, j]):
                            mark_mat[i, j] = 1

        nan_mat = np.isnan(mview_kps2d[..., 0, 0])
        mark_mat[nan_mat] = 0
        update_idx = np.array(range(n_cameras)).tolist()
        return_matrix = np.ones_like(nan_mat) * -1
        counting_idx = np.where(~nan_mat)
        for count, (i, j) in enumerate(zip(counting_idx[0], counting_idx[1])):
            return_matrix[i, j] = count

        matched = []
        matched_list = []
        record_error = []

        matching = build_matching(self.cfg.advance_sort_tracking.matching)

        # Anchor camera based matching
        for idx, anchor in enumerate(homo_points[0]):

            if mark_mat[0, idx] > 0 or idx > 0 and nan_mat[0, idx - 1] > 0:
                continue
            tri_human_idx = []
            tri_kps2d = []
            n_tri_cameras = []

            result = matching(
                mview_kps2d,
                mark_mat,
                nan_mat, [idx], [mview_kps2d[0, idx]],
                homo_thres=200,
                homo_anchor=anchor,
                points=homo_points)
            if len(result) > 0:
                tri_human_idx += result[0]
                tri_kps2d += result[1]
                n_tri_cameras += result[2]

            if len(tri_human_idx) == 0:
                continue
            tri_human_idx = np.stack(tri_human_idx, axis=0)  # (N, n_cam)
            tri_kps2d = np.stack(tri_kps2d, axis=0)  # (N, n_cam, n_kps2d, 2)

            best_idx = -1
            best_error = 1000
            best_record_error = 1000
            best_kps3d = None

            # Iterate through all the possible matching
            for index, (kps2d, _) in enumerate(zip(tri_kps2d, tri_human_idx)):

                kps3d = triangulator.triangulate(kps2d)
                error = triangulator.get_reprojection_error(
                    points2d=kps2d, points3d=kps3d, mean=True)
                if np.isnan(error).all():
                    continue
                error = error[~np.isnan(error)].mean(
                ) / n_tri_cameras[index]**n_cameras

                if error < best_error and error < self.match_2d_2d_thres:
                    best_idx = index
                    best_error = error
                    best_kps3d = kps3d

            if best_kps3d is not None:
                if self.verbose:
                    logger.info(f'2d-2d matching min error: {best_error}')
                record_error.append(best_record_error)
                picked_idx = tri_human_idx[best_idx]
                new_kps2d = mview_kps2d[update_idx, picked_idx]
                new_track_id = track_id[update_idx, picked_idx]
                new_human = Human(
                    frame_id,
                    new_kps2d,
                    best_kps3d,
                    new_track_id,
                    convention=convention)
                if new_human.is_human(
                        self.always_is_human,
                        human_bone_thres=self.human_bone_thresholds):
                    matched.append(new_human)
                    mark_mat[update_idx, picked_idx] = 1
                    mark_mat[nan_mat] = 0
                    raw = return_matrix[update_idx, picked_idx]
                    if self.gt_label:
                        matched_list.append(new_track_id.tolist())
                    else:
                        matched_list.append(raw[raw > -1].tolist())
        return matched_list, matched, record_error

    def associate_detections_to_human(
        self,
        humans: List[Human],
        data: tuple,
        triangulator,
        detection_mask: Union[None, np.ndarray] = None,
        use_homo: bool = True,
        logger: Union[None, str, logging.Logger] = None
    ) -> Tuple[list, list, np.ndarray, list, list]:
        """Implementation for 2d-3d association.

        Args:
            humans (List[Human]): Multiple active Human objects
            data (tuple): The input data, including H projected points,
                2d keypoints, track_id and frame id
            triangulator (AniposelibTriangulator): AniposelibTriangulator.
            detection_mask (Union[None, np.ndarray], optional): The 2d detected
                mask, 1 indicates matched observations. Defaults to None.
            use_homo (bool, optional): Whether to use homography related
                operation for reduced graph. Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                    Logger for logging. If None, root logger will be selected.
                    Defaults to None.

        Returns:
            Tuple[list, list, np.ndarray, list, list]:
            matched_list: The index of matched person
            matched: Human object
            mark_mat: The 2d detection mask indicating unmatched observations
            unmatched: unmatched person
            record_error: total error
        """
        logger = get_logger(logger)
        # Error weightage hyperparameters
        reprojection_weight = 5

        points, mview_kps2d, track_id, frame_id = data
        n_cameras, n_human, _, _ = mview_kps2d.shape
        # Instantiate the matrix
        if detection_mask is not None:
            mark_mat = detection_mask
        else:
            mark_mat = np.zeros_like(mview_kps2d[:, :, 0, 0])
            # Filter homo projection based on field dimension
            if use_homo:
                for i in range(n_cameras):
                    for j in range(n_human):
                        if out_of_field(points[i, j]):
                            mark_mat[i, j] = 1

        nan_mat = np.isnan(mview_kps2d[..., 0, 0])
        mark_mat[nan_mat] = 0
        update_idx = np.array(range(n_cameras)).tolist()
        return_matrix = np.ones_like(nan_mat) * -1
        counting_idx = np.where(~nan_mat)
        for count, (i, j) in enumerate(zip(counting_idx[0], counting_idx[1])):
            return_matrix[i, j] = count

        matched = []
        unmatched = []
        matched_list = []
        record_error = []

        matching = build_matching(self.cfg.advance_sort_tracking.matching)

        # Anchor human based matching
        for human in humans:
            tri_kps2d = []
            tri_human_idx = []
            n_tri_cameras = []
            homo_anchor, homo_thres = human.get_homo_threshold()
            # Build the matching

            result = matching(
                mview_kps2d,
                mark_mat,
                nan_mat, [], [],
                homo_thres=homo_thres,
                homo_anchor=homo_anchor,
                points=points)
            if len(result) > 0:
                tri_human_idx += result[0]
                tri_kps2d += result[1]
                n_tri_cameras += result[2]
            if len(tri_human_idx) == 0:
                unmatched.append(human)
                continue

            tri_human_idx = np.stack(tri_human_idx, axis=0)  # (N, n_cameras)
            tri_kps2d = np.stack(tri_kps2d, axis=0)  # (N, n_cameras, n_kps, 2)

            best_idx = -1
            best_error = 1000
            best_record_error = 1000
            best_kps3d = None

            # Iterate through all the possible matching
            for index, (kps2d,
                        human_idx) in enumerate(zip(tri_kps2d, tri_human_idx)):
                kps3d = triangulator.triangulate(kps2d)
                error = triangulator.get_reprojection_error(
                    points2d=kps2d, points3d=kps3d, mean=True)
                if np.isnan(error).all():
                    continue
                proj_error = error[~np.isnan(error)].mean(
                ) / n_tri_cameras[index]**n_cameras
                kps3d_error = human.error_key_p3(kps3d)
                track_error = human.error_track(track_id[update_idx,
                                                         human_idx])
                welsch_error = human.error_welsch(n_tri_cameras[index],
                                                  n_cameras)
                total_error = reprojection_weight * proj_error +\
                    self.kps3d_weight * kps3d_error + self.track_weight *\
                    track_error + self.welsch_weight * welsch_error

                if total_error < best_error and \
                   total_error < self.match_2d_3d_thres:
                    best_idx = index
                    best_error = total_error
                    best_record_error = total_error
                    best_kps3d = kps3d

            if best_kps3d is not None:
                if self.verbose:
                    logger.info(f'2d-3d matching min error: {best_error}')
                record_error.append(best_record_error)
                picked_idx = tri_human_idx[best_idx]
                new_kps2d = mview_kps2d[update_idx, picked_idx]
                new_track_id = track_id[update_idx, picked_idx]
                human.update(new_kps2d, best_kps3d, new_track_id)

                human.kalman_update(best_kps3d)
                mark_mat[update_idx, picked_idx] = 1
                mark_mat[nan_mat] = 0
                raw = return_matrix[update_idx, picked_idx]
                matched.append(human)
                if self.gt_label:
                    matched_list.append(new_track_id.tolist())
                else:
                    matched_list.append(raw[raw > -1].tolist())
            else:
                unmatched.append(human)

        return matched_list, matched, mark_mat, unmatched, record_error
