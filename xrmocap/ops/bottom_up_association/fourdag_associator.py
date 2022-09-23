# yapf: disable
from asyncio.log import logger
import logging
import numpy as np
from tkinter import NO
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.bottom_up_association.associate.builder import build_associate
from xrmocap.ops.top_down_association.identity_tracking.builder import (
    BaseTracking, build_identity_tracking,
)
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, build_point_selector,
)
from xrmocap.transform.keypoints3d.optim.builder import (
    build_keypoints3d_optimizer,
)
from xrmocap.utils.fourdag_utils import skel_info

# yapf: enable


class FourDAGAssociator:

    def __init__(self,
                 kps_convention: str = 'fourdag_19',
                 triangulator: Union[None, dict, BaseTriangulator] = None,
                 point_selector: Union[None, dict, BaseSelector] = None,
                 keypoints3d_optimizer=None,
                 associate_graph: Union[None, dict] = None,
                 identity_tracking: Union[None, dict, BaseTracking] = None,
                 min_asgn_cnt: int = 5,
                 use_tracking_edges: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:

        self.logger = get_logger(logger)

        if isinstance(triangulator, dict):
            triangulator['logger'] = self.logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        if isinstance(keypoints3d_optimizer, dict):
            keypoints3d_optimizer['logger'] = self.logger
            self.keypoints3d_optimizer = build_keypoints3d_optimizer(
                keypoints3d_optimizer)
        else:
            self.keypoints3d_optimizer = keypoints3d_optimizer

        self.n_views = -1
        self.kps_convention = kps_convention
        self.last_multi_kps3d = dict()
        self.use_tracking_edges = use_tracking_edges
        self.min_asgn_cnt = min_asgn_cnt
        if isinstance(point_selector, dict):
            point_selector['logger'] = self.logger
            self.point_selector = build_point_selector(point_selector)
        else:
            self.point_selector = point_selector
        if isinstance(associate_graph, dict):
            associate_graph['logger'] = self.logger
            associate_graph['n_kps'] = skel_info[self.kps_convention]['n_kps']
            associate_graph['n_pafs'] = skel_info[
                self.kps_convention]['n_pafs']
            self.associate_graph = build_associate(associate_graph)
        else:
            self.associate_graph = associate_graph
        if isinstance(identity_tracking, dict):
            identity_tracking['logger'] = self.logger
            self.identity_tracking = build_identity_tracking(identity_tracking)
        else:
            self.identity_tracking = identity_tracking

    def set_cameras(
        self, cameras: List[Union[FisheyeCameraParameter,
                                  PinholeCameraParameter]]
    ) -> None:
        if self.triangulator is not None:
            self.triangulator.set_cameras(cameras)
        if self.keypoints3d_optimizer is not None:
            self.keypoints3d_optimizer.set_cameras(cameras)
        if hasattr(self.point_selector, 'triangulator'):
            self.point_selector.triangulator.set_cameras(cameras)
        self.associate_graph.set_cameras(cameras)
        self.n_views = len(cameras)

    def cal_keypoints2d(self, m_persons_map, kps2d):
        for i, person_id in enumerate(m_persons_map.copy()):
            if i < len(self.last_multi_kps3d):
                continue
            if sum(sum(m_persons_map[person_id] >= 0)) >= self.min_asgn_cnt:
                continue
            else:
                m_persons_map.pop(person_id)

        m_skels2d = {}
        
        for person_id in m_persons_map:
            if person_id < len(self.last_multi_kps3d):
                identity = list(self.last_multi_kps3d.keys())[person_id] ###change 
            elif len(m_skels2d) == 0:
                identity = 0
            else:
                identity = max(m_skels2d) + 1
            skel2d = np.zeros((3, self.n_views * self.n_kps))
            for view in range(self.n_views):
                for joint_id in range(self.n_kps):
                    index = m_persons_map[person_id][joint_id, view]
                    if index != -1:
                        skel2d[:, view * self.n_kps +
                               joint_id] = kps2d[view][joint_id][index]
                    else:
                        continue
            m_skels2d[identity] = skel2d
        return m_skels2d

    def associate_frame(self,
                        kps2d: list,
                        pafs: list,
                        end_of_clip=False) -> Tuple[Keypoints, List[int]]:
        """Associate and triangulate keypoints2d in one frame.

        Args:

        Returns:

            keypoints3d (Keypoints):
                An instance of class keypoints,
                triangulated from the selected
                keypoints2d.
            indentities (List[int]):
                A list of indentities, whose length.
        """
        if end_of_clip:
            self.last_multi_kps3d = dict()
        self.n_kps = len(kps2d[0])
        m_persons_map = self.associate_graph(kps2d, pafs,
                                             self.last_multi_kps3d)
        self.logger.info('m_persons_map:{}'.format(list(m_persons_map.keys())))
        m_skels2d = self.cal_keypoints2d(m_persons_map, kps2d)

        if self.keypoints3d_optimizer is not None:
            self.logger.info('m_skels2d:{}'.format(list(m_skels2d.keys())))
            multi_kps3d = self.keypoints3d_optimizer.update(m_skels2d)
            self.logger.info('multi_kps3d:{}'.format(list(multi_kps3d.keys())))
            if self.use_tracking_edges:
                self.last_multi_kps3d = multi_kps3d
            kps_arr = np.zeros((1, len(multi_kps3d), self.n_kps, 4))
            mask_arr = np.zeros((1, len(multi_kps3d), self.n_kps))
            for index, pid in enumerate(multi_kps3d):
                kps_arr[0, index, ...] = multi_kps3d[pid][:, :self.n_kps].T
                mask_arr[0, index, :] = multi_kps3d[pid][3, :self.n_kps]
            keypoints3d = Keypoints(
                kps=kps_arr, mask=mask_arr, convention=self.kps_convention)
            identities = multi_kps3d.keys()

            multi_kps2d = []
            for person_id in m_skels2d:
                kps2d = np.zeros((self.n_views, self.n_kps, 3))
                for view in range(self.n_views):
                    for joint_id in range(self.n_kps):
                        kps2d[view][joint_id] = m_skels2d[
                            person_id][:, view * self.n_kps + joint_id]
                multi_kps2d.append(kps2d)
            multi_kps2d = np.array(multi_kps2d)

        elif self.triangulator is not None:
            multi_kps3d = []
            identities = []
            multi_kps2d = []
            for person_id in m_skels2d:
                kps2d = np.zeros((self.n_views, self.n_kps, 3))
                for view in range(self.n_views):
                    for joint_id in range(self.n_kps):
                        kps2d[view][joint_id] = m_skels2d[
                            person_id][:, view * self.n_kps + joint_id]
                multi_kps2d.append(kps2d)
                matched_mkps2d = np.zeros((self.n_views, self.n_kps, 2))
                matched_mkps2d_mask = np.zeros((self.n_views, self.n_kps, 1))
                matched_mkps2d_conf = np.zeros((self.n_views, self.n_kps, 1))
                matched_mkps2d = kps2d[..., :2]
                matched_mkps2d_mask = np.ones_like(kps2d[..., 0:1])
                matched_mkps2d_conf[..., 0] = kps2d[..., 2]
                selected_mask = self.point_selector.get_selection_mask(
                    np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                   axis=-1), matched_mkps2d_mask)
                kps3d = self.triangulator.triangulate(matched_mkps2d,
                                                      selected_mask)

                if not np.isnan(kps3d).all():
                    multi_kps3d.append(kps3d)
            multi_kps3d = np.array(multi_kps3d)
            multi_kps2d = np.array(multi_kps2d)

            if len(multi_kps3d) > 0:
                keypoints3d, identities = self.assign_identities_frame(
                    multi_kps3d)
                self.last_multi_kps3d = dict()
                if self.use_tracking_edges:
                    for index, pid in enumerate(identities):
                        self.last_multi_kps3d[pid] = keypoints3d.get_keypoints(
                        )[0, index, ...].T
            else:
                keypoints3d = Keypoints()
                identities = []
                self.last_multi_kps3d = dict()

        return keypoints3d, identities, multi_kps2d, m_persons_map

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
