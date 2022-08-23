# yapf: disable
import logging
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from PIL import Image
from torchvision.transforms import transforms as T
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.ops.top_down_association.body_tracking.builder import (
    build_kalman_tracking,
)
from xrmocap.ops.top_down_association.matching.builder import build_matching
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, build_point_selector,
)
from .identity_tracking.builder import BaseTracking, build_identity_tracking

# yapf: enable


class MvposeAssociator:

    def __init__(self,
                 triangulator: Union[None, dict, BaseTriangulator],
                 affinity_estimator: Union[None, dict, BaseSelector] = None,
                 point_selector: Union[None, dict, BaseSelector] = None,
                 multi_way_matching: Union[None, dict] = None,
                 kalman_tracking: Union[None, dict] = None,
                 identity_tracking: Union[None, dict, BaseTracking] = None,
                 checkpoint_path: str = None,
                 best_distance: int = 600,
                 interval: int = 5,
                 bbox_thr: float = 0.9,
                 device: str = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:

        self.logger = get_logger(logger)

        if isinstance(triangulator, dict):
            triangulator['logger'] = self.logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        if isinstance(affinity_estimator, dict):
            self.affinity_estimator = build_architecture(affinity_estimator)
        else:
            self.affinity_estimator = affinity_estimator

        if isinstance(point_selector, dict):
            point_selector['logger'] = self.logger
            self.point_selector = build_point_selector(point_selector)
        else:
            self.point_selector = point_selector

        if isinstance(multi_way_matching, dict):
            multi_way_matching['logger'] = self.logger
            self.multi_way_matching = build_matching(multi_way_matching)
        else:
            self.multi_way_matching = multi_way_matching

        if isinstance(identity_tracking, dict):
            identity_tracking['logger'] = self.logger
            self.identity_tracking = build_identity_tracking(identity_tracking)
        else:
            self.identity_tracking = identity_tracking

        if isinstance(kalman_tracking, dict):
            kalman_tracking['logger'] = self.logger
            self.kalman_tracking = build_kalman_tracking(kalman_tracking)
        else:
            self.kalman_tracking = kalman_tracking

        if checkpoint_path is not None:
            load_checkpoint(
                self.affinity_estimator,
                checkpoint_path,
                map_location=device,
                logger=logger)
        # save the config in the model for convenience
        self.affinity_estimator.to(device)
        self.affinity_estimator.eval()
        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.test_transformer = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.ToTensor(),
            self.normalizer,
        ])
        self.best_distance = best_distance
        self.interval = interval
        self.bbox_thr = bbox_thr
        self.counter = 0
        self.device = device
        self.n_views = -1
        self.fundamental_mat = None
        self.last_multi_kps3d = None

    def set_cameras(
        self, cameras: List[Union[FisheyeCameraParameter,
                                  PinholeCameraParameter]]
    ) -> None:
        self.cameras = cameras
        self.triangulator.set_cameras(cameras)
        if hasattr(self.point_selector, 'triangulator'):
            self.point_selector.triangulator.set_cameras(cameras)
        self.n_views = len(cameras)
        self.fundamental_mat = self.calc_fundamental_mat()

    def associate_frame(
            self, mview_img_arr: np.ndarray, mview_bbox2d: List[torch.Tensor],
            mview_keypoints2d: List[Keypoints], affinity_type: str
    ) -> Tuple[List[List[int]], Keypoints, List[int]]:
        """Associate and triangulate keypoints2d in one frame.

        Args:
            mview_img_arr (np.ndarray):
                Multi-view image array, in shape
                [n_frames, h, w, ch].
            mview_bbox2d (List[torch.Tensor]):
                Multi-view bbox2d.
            mview_keypoints2d (List[Keypoints]):
                A list of multi-view keypoints2d,
                detected from the mview_img_arr.
                len(mview_keypoints2d) == n_frames.
            affinity_type (str):
                The expression of geometric affinity and appearance affinity

        Returns:
            association_results (List[List[int]]):
                A nested list of association result,
                in shape [n_person, n_view], and
                association_results[i][j] = k means
                the k-th 2D perception in view j
                is a 2D obersevation of person i.
            keypoints3d (Keypoints):
                An instance of class keypoints,
                triangulated from the selected
                keypoints2d.
            indentities (List[int]):
                A list of indentities, whose length
                is equal to len(association_results).
        """
        multi_kps3d = []
        multi_tracking_kps3d = []
        multi_kps2d_idx = []
        mview_person_id = []
        kalman_tracking_requires_init = False
        not_matched_idx = None
        self.kps_convention = mview_keypoints2d[0].get_convention()
        for bbox2d_i in mview_bbox2d:
            person_id = np.array([
                i for i, data in enumerate(bbox2d_i)
                if data[-1] > self.bbox_thr
            ])
            mview_person_id.append(person_id)
        image_tensor, kps2d, dim_group, n_kps2d, bbox2d = self.process_data(
            mview_person_id, mview_img_arr, mview_bbox2d, mview_keypoints2d)
        image_tensor = image_tensor.to(self.device)
        if self.kalman_tracking is not None:
            if self.counter == 0:
                kalman_tracking_requires_init = True
            else:
                self.kalman_tracking.predict()
                multi_tracking_kps2d_idx, multi_tracking_kps3d,\
                    not_matched_idx, not_matched_dim_group =\
                    self.tracking_frame(kps2d, dim_group, n_kps2d, bbox2d)
                if len(not_matched_idx) > 0:
                    kps2d = kps2d[not_matched_idx]
                else:
                    kps2d = np.array([])
                dim_group = not_matched_dim_group
            self.counter += 1
            if self.counter == self.interval:
                self.counter = 0

        if len(kps2d) > 0:  # detect the person
            kps2d_conf = kps2d[..., 2:3]
            kps2d = kps2d[..., :2]
            matched_list, sub_imgid2cam = self.multi_way_matching(
                kps2d, image_tensor, self.affinity_estimator,
                self.fundamental_mat, affinity_type, n_kps2d, dim_group,
                not_matched_idx)
            mkps2d_id = np.zeros(sub_imgid2cam.shape, dtype=np.int32)
            for i in range(self.n_views):
                for j in range(dim_group[i + 1] - dim_group[i]):
                    mkps2d_id[dim_group[i] + j] = j
            for person in matched_list:
                if person.shape[0] < 2:
                    continue
                matched_mkps2d_id = np.full(self.n_views, np.nan)
                matched_mkps2d_id[sub_imgid2cam[person]] = mkps2d_id[person]
                matched_mkps2d = np.zeros((self.n_views, n_kps2d, 2))
                matched_mkps2d_mask = np.zeros((self.n_views, n_kps2d, 1))
                matched_mkps2d_conf = np.zeros((self.n_views, n_kps2d, 1))

                matched_mkps2d[sub_imgid2cam[person]] = kps2d[person]
                matched_mkps2d_mask[sub_imgid2cam[person]] = np.ones_like(
                    kps2d[person][..., 0:1])
                matched_mkps2d_conf[sub_imgid2cam[person]] = kps2d_conf[person]

                selected_mask = self.point_selector.get_selection_mask(
                    np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                   axis=-1), matched_mkps2d_mask)
                kps3d = self.triangulator.triangulate(matched_mkps2d,
                                                      selected_mask)
                if not np.isnan(kps3d).all():
                    multi_kps3d.append(kps3d)
                    multi_kps2d_idx.append(matched_mkps2d_id)
            multi_kps3d = np.array(multi_kps3d)
        if len(multi_tracking_kps3d) > 0:
            if len(multi_kps3d) > 0:
                multi_kps3d = np.concatenate(
                    (multi_tracking_kps3d, multi_kps3d), axis=0)
                multi_tracking_kps2d_idx.extend(multi_kps2d_idx)
                kalman_tracking_requires_init = True
            else:
                multi_kps3d = multi_tracking_kps3d
            multi_kps2d_idx = multi_tracking_kps2d_idx
        if len(multi_tracking_kps3d) == 0 and len(multi_kps3d) > 0:
            kalman_tracking_requires_init = True

        if self.kalman_tracking is not None and kalman_tracking_requires_init:
            self.kalman_tracking.set_init_kps3d(state_kps3d=multi_kps3d)
        if len(multi_kps3d) > 0:
            keypoints3d, identities = self.assign_identities_frame(multi_kps3d)
            self.last_multi_kps3d = keypoints3d.get_keypoints()[0, ..., :3]
        else:
            keypoints3d = Keypoints()
            identities = []
            self.counter = 0
            kalman_tracking_requires_init = False

        return multi_kps2d_idx, keypoints3d, identities

    def tracking_frame(
        self,
        kps2d,
        dim_group,
        n_kps2d,
        bbox2d,
    ) -> Tuple[List[List[int]], Keypoints, List[int]]:
        multi_kps3d = []
        multi_kps2d_idx = []

        sub_imgid2cam = np.zeros(kps2d.shape[0], dtype=np.int32)
        for idx, i in enumerate(range(self.n_views)):
            sub_imgid2cam[dim_group[i]:dim_group[i + 1]] = idx
        kps2d_conf = kps2d[..., 2:3]
        kps2d = kps2d[..., :2]
        mkps2d_id = np.zeros(sub_imgid2cam.shape, dtype=np.int32)
        for i in range(self.n_views):
            for j in range(dim_group[i + 1] - dim_group[i]):
                mkps2d_id[dim_group[i] + j] = j

        measurement_kps3d, multi_kps2d_idx, not_matched_dim_group,\
            not_matched_idx = self.kalman_tracking.get_measurement_kps3d(
                self.n_views,
                self.point_selector,
                self.last_multi_kps3d,
                sub_imgid2cam,
                kps2d,
                kps2d_conf,
                n_kps2d,
                self.best_distance,
                self.triangulator,
                mkps2d_id,
                tracking_bbox=bbox2d)
        multi_kps3d = self.kalman_tracking.update(measurement_kps3d)
        return multi_kps2d_idx, multi_kps3d, not_matched_idx,\
            not_matched_dim_group

    def process_data(self, mview_person_id, mview_img_arr, mview_bbox2d,
                     mview_keypoints2d):
        cropped_img = []
        kps2d = []
        ret_bbox2d = []
        cnt = 0
        this_dim = [0]
        for view, person_id in enumerate(mview_person_id):
            v_bbox2d = mview_bbox2d[view][person_id]
            ret_bbox2d.append(v_bbox2d.numpy())
            img = mview_img_arr[view]
            v_bbox2d[v_bbox2d < 0] = 0
            v_cropped_img = [
                img[:,
                    int(bbox2d[1]):int(bbox2d[3]),
                    int(bbox2d[0]):int(bbox2d[2])] for bbox2d in v_bbox2d
            ]
            for data in v_cropped_img:
                cropped_img.append(
                    self.test_transformer(
                        Image.fromarray(data.transpose(1, 2, 0))))
            if len(person_id) > 0:
                kps2d.append(
                    mview_keypoints2d[view].get_keypoints()[0][person_id])

            n_person = len(person_id)
            cnt += n_person
            this_dim.append(cnt)
        dim_group = torch.Tensor(this_dim).long()
        cropped_img = torch.stack(cropped_img)
        kps2d = np.concatenate(kps2d, axis=0)
        n_kps2d = mview_keypoints2d[0].get_keypoints_number()
        ret_bbox2d = np.concatenate(ret_bbox2d, axis=0)

        return cropped_img, kps2d, dim_group, n_kps2d, ret_bbox2d

    def calc_fundamental_mat(self, cam_world2cam=True):
        camera_parameter = {
            'K': np.zeros((self.n_views, 3, 3)),
            'RT': np.zeros((self.n_views, 3, 4)),
        }
        for i, input_cam_param in enumerate(self.cameras):
            cam_param = input_cam_param.clone()
            if cam_param.world2cam != cam_world2cam:
                cam_param.inverse_extrinsic()
            camera_parameter['K'][i] = cam_param.get_intrinsic(k_dim=3)
            camera_parameter['RT'][i] = np.concatenate(
                (np.asarray(cam_param.extrinsic_r),
                 np.asarray(cam_param.extrinsic_t)[:, np.newaxis]),
                axis=1)
        K = camera_parameter['K'].astype(np.float32)
        RT = camera_parameter['RT'].astype(np.float32)
        # calculate the fundamental matrix for geometry affinity
        self.skew_op = lambda x: torch.tensor([[0, -x[2], x[
            1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

        self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1:\
            torch.inverse(K_0).t() @ (R_0 @ R_1.t()) @ K_1.t() @ \
            self.skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))

        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1:\
            self.fundamental_op(
                K_0, RT_0[:, :3], RT_0[:, 3], K_1, RT_1[:, :3], RT_1[:, 3])

        ret_mat = torch.zeros(self.n_views, self.n_views, 3, 3)
        for i in range(self.n_views):
            for j in range(self.n_views):
                ret_mat[i, j] += self.fundamental_RT_op(
                    torch.tensor(K[i]), torch.tensor(RT[i]),
                    torch.tensor(K[j]), torch.tensor(RT[j]))
                if ret_mat[i, j].sum() == 0:
                    ret_mat[i, j] += 1e-12  # to avoid nan
        return ret_mat

    def assign_identities_frame(self, curr_kps3d) -> Keypoints:
        """Process kps3d to Keypoints (an instance of class Keypoints,
        including kps data, mask and convention).

        Args:
            curr_kps3d (List[np.ndarray]): The results of each frame.

        Returns:
            Keypoints: An instance of class Keypoints.
        """
        frame_identity = self.identity_tracking.query(curr_kps3d)

        kps3d_score = np.ones_like(curr_kps3d[..., 0:1])
        kps3d = (np.concatenate((curr_kps3d, kps3d_score), axis=-1))
        kps3d = kps3d[np.newaxis]
        kps3d_mask = np.ones_like(kps3d[..., 0])
        keypoints3d = Keypoints(kps=kps3d, convention=self.kps_convention)
        keypoints3d.set_mask(kps3d_mask)

        return keypoints3d, frame_identity
