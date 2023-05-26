# yapf: disable
import logging
import numpy as np
from typing import List, Tuple, Union
from xrprimer.data_structure import Keypoints
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger

from xrmocap.ops.bottom_up_association.graph_solver.builder import (
    build_graph_solver,
)
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

# yapf: enable


class FourDAGAssociator:

    def __init__(self,
                 kps_convention: str = 'fourdag_19',
                 triangulator: Union[None, dict, BaseTriangulator] = None,
                 point_selector: Union[None, dict, BaseSelector] = None,
                 keypoints3d_optimizer=None,
                 n_views: int = 5,
                 graph_construct: Union[None, dict] = None,
                 graph_associate: Union[None, dict] = None,
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

        self.n_views = n_views
        self.kps_convention = kps_convention
        self.last_multi_kps3d = dict()
        self.use_tracking_edges = use_tracking_edges
        self.min_asgn_cnt = min_asgn_cnt
        if isinstance(point_selector, dict):
            point_selector['logger'] = self.logger
            self.point_selector = build_point_selector(point_selector)
        else:
            self.point_selector = point_selector
        if isinstance(graph_associate, dict):
            graph_associate['logger'] = self.logger
            graph_associate['n_views'] = self.n_views
            self.graph_associate = build_graph_solver(graph_associate)
        else:
            self.graph_associate = graph_associate
        if isinstance(graph_construct, dict):
            graph_construct['logger'] = self.logger
            graph_construct['n_views'] = self.n_views
            self.graph_construct = build_graph_solver(graph_construct)
        else:
            self.graph_construct = graph_construct
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
        self.graph_construct.set_cameras(cameras)

    def cal_keypoints2d(self, mpersons_map, kps2d):
        for i, person_id in enumerate(mpersons_map.copy()):
            if person_id in self.last_multi_kps3d:
                continue
            if sum(sum(mpersons_map[person_id] >= 0)) >= self.min_asgn_cnt:
                continue
            else:
                mpersons_map.pop(person_id)

        m_limbs2d = {}
        for person_id in mpersons_map:
            if person_id in self.last_multi_kps3d:
                identity = person_id
            elif len(m_limbs2d) == 0:
                identity = 0
            else:
                identity = max(m_limbs2d) + 1
            limb2d = np.zeros((3, self.n_views * self.n_kps))
            for view in range(self.n_views):
                for kps_id in range(self.n_kps):
                    index = mpersons_map[person_id][kps_id, view]
                    if index != -1:
                        limb2d[:, view * self.n_kps +
                               kps_id] = kps2d[view][kps_id][index]
            m_limbs2d[identity] = limb2d
        return m_limbs2d

    def associate_frame(self,
                        kps2d: list,
                        pafs: list,
                        end_of_clip=False) -> Tuple[Keypoints, List[int]]:
        """Associate and triangulate keypoints2d in one frame.

        Args:
            kps2d (List):
                data for bottom-up keypoints in shape
                    [n_views, n_kps, n_candidatas, 3]
            pafs (List):
                data for pafs in shape
                    [n_views, n_pafs, n_candidatas_1, n_candidatas_2]
            end_of_clip (bool):
                indicator of end of a clip
        Returns:

            keypoints3d (Keypoints):
                An instance of class keypoints,
                triangulated from the selected
                keypoints2d.
            indentities (List[int]):
                A list of indentities, whose length
                    represets the number of person.
            multi_kps2d:
                the associated keypoints
            mpersons_map:
                the associated maps
        """

        self.n_kps = len(kps2d[0])
        graph_info = self.graph_construct(kps2d, pafs, self.last_multi_kps3d)
        mpersons_map = self.graph_associate(kps2d, pafs, graph_info,
                                            self.last_multi_kps3d)
        mlimbs2d = self.cal_keypoints2d(mpersons_map, kps2d)
        multi_kps2d = dict()
        for person_id in mlimbs2d:
            mview_kps2d = np.zeros((self.n_views, self.n_kps, 3))
            for view in range(self.n_views):
                for kps_id in range(self.n_kps):
                    mview_kps2d[view][kps_id] = mlimbs2d[
                        person_id][:, view * self.n_kps + kps_id]
            multi_kps2d[person_id] = mview_kps2d

        if self.keypoints3d_optimizer is not None:
            multi_kps3d = self.keypoints3d_optimizer.update(mlimbs2d)
            if self.use_tracking_edges:
                self.last_multi_kps3d = multi_kps3d
            kps_arr = np.zeros((1, len(multi_kps3d), self.n_kps, 4))
            mask_arr = np.zeros((1, len(multi_kps3d), self.n_kps))
            for index, person_id in enumerate(multi_kps3d):
                kps_arr[0, index,
                        ...] = multi_kps3d[person_id][:, :self.n_kps].T
                mask_arr[0, index, :] = multi_kps3d[person_id][3, :self.n_kps]
            keypoints3d = Keypoints(
                kps=kps_arr, mask=mask_arr, convention=self.kps_convention)
            identities = list(multi_kps3d.keys())
        elif self.triangulator is not None:
            multi_kps3d = []
            identities = []
            for person_id in mlimbs2d:
                mview_kps2d = multi_kps2d[person_id]
                matched_mkps2d = np.zeros((self.n_views, self.n_kps, 2))
                matched_mkps2d_mask = np.zeros((self.n_views, self.n_kps, 1))
                matched_mkps2d_conf = np.zeros((self.n_views, self.n_kps, 1))
                matched_mkps2d = mview_kps2d[..., :2]
                matched_mkps2d_mask = np.ones_like(mview_kps2d[..., 0:1])
                matched_mkps2d_conf[..., 0] = mview_kps2d[..., 2]
                selected_mask = self.point_selector.get_selection_mask(
                    np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                   axis=-1), matched_mkps2d_mask)
                kps3d = self.triangulator.triangulate(matched_mkps2d,
                                                      selected_mask)

                if not np.isnan(kps3d).all():
                    multi_kps3d.append(kps3d)
                    identities.append(person_id)
            multi_kps3d = np.array(multi_kps3d)
            kps3d_score = np.ones_like(multi_kps3d[..., 0:1])
            kps3d = (np.concatenate((multi_kps3d, kps3d_score), axis=-1))
            kps3d = kps3d[np.newaxis]
            kps3d_mask = np.ones_like(kps3d[..., 0])
            keypoints3d = Keypoints(kps=kps3d, convention=self.kps_convention)
            keypoints3d.set_mask(kps3d_mask)
            if self.use_tracking_edges:
                for index, person_id in enumerate(identities):
                    self.last_multi_kps3d[
                        person_id] = keypoints3d.get_keypoints()[0, index,
                                                                 ...].T
        if end_of_clip:
            self.last_multi_kps3d = dict()
            if self.keypoints3d_optimizer is not None:
                self.keypoints3d_optimizer.trace_limbs.clear()
                self.keypoints3d_optimizer.trace_limb_infos.clear()
        return keypoints3d, identities, multi_kps2d, mpersons_map

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
