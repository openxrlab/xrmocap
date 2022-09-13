# yapf: disable
import logging
import numpy as np
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.bottom_up_association.matching.builder import build_matching
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, build_point_selector,
)
from xrmocap.ops.top_down_association.identity_tracking.builder import BaseTracking, build_identity_tracking
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_idx,
)
# yapf: enable

class FourDAGAssociator:

    def __init__(self,
                 kps_convention: str,
                 triangulator: Union[None, dict, BaseTriangulator],
                 point_selector: Union[None, dict, BaseSelector] = None,
                 fourd_matching: Union[None, dict] = None,
                 identity_tracking: Union[None, dict, BaseTracking] = None,
                 min_asgn_cnt: int = 5 ,
                 logger: Union[None, str, logging.Logger] = None) -> None:

        self.logger = get_logger(logger)

        if isinstance(triangulator, dict):
            triangulator['logger'] = self.logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator
        
        self.n_views = -1
        self.kps_convention = kps_convention
        self.last_multi_kps3d = dict()
        self.min_asgn_cnt = min_asgn_cnt
        if isinstance(point_selector, dict):
            point_selector['logger'] = self.logger
            self.point_selector = build_point_selector(point_selector)
        else:
            self.point_selector = point_selector
        if isinstance(fourd_matching, dict):
            fourd_matching['logger'] = self.logger
            self.fourd_matching = build_matching(fourd_matching)
        else:
            self.fourd_matching = fourd_matching
        if isinstance(identity_tracking, dict):
            identity_tracking['logger'] = self.logger
            self.identity_tracking = build_identity_tracking(identity_tracking)
        else:
            self.identity_tracking = identity_tracking

    def set_cameras(
        self, cameras: List[Union[FisheyeCameraParameter,
                                  PinholeCameraParameter]]
    ) -> None:
        
        self.triangulator.set_cameras(cameras)
        if hasattr(self.point_selector, 'triangulator'):
            self.point_selector.triangulator.set_cameras(cameras)
        self.fourd_matching.set_cameras(cameras)
        self.n_views = len(cameras)


    def cal_keypoints2d(self, m_personsMap, kps2d_paf):
    
        for i, person_id in enumerate(m_personsMap.copy()):
            if i < len(self.last_multi_kps3d):
                continue
            if  sum(sum(m_personsMap[person_id] >= 0)) >= self.min_asgn_cnt:
                continue
            else:
                m_personsMap.pop(person_id)

        m_skels2d = {}
        for person_id in m_personsMap:
            if person_id  < len(self.last_multi_kps3d):
                identity = person_id
            elif len(m_skels2d) == 0:
                identity = 0
            else:
                identity = max(m_skels2d) + 1
            skel2d = np.zeros((3,self.n_views *self.n_kps))
            for view in range(self.n_views):
                for joint_id in range(self.n_kps):
                    index = m_personsMap[person_id][joint_id, view]
                    if index != -1:
                        skel2d[:,view*self.n_kps+joint_id] = kps2d_paf[view]['joints'][joint_id][index]
                    else:
                        continue

            m_skels2d[identity] = skel2d
        return m_skels2d
    
    def MappingToSkel17(self,skel19):
        keypoints2d = np.zeros((self.n_views, 19, 3))
        for view in range(self.n_views):
            for joint_id in range(19):
                keypoints2d[view][joint_id] = skel19[:,view*self.n_kps+joint_id]
        shelf17 = np.zeros((self.n_views, 17, 3), dtype=np.float)
        mapping = [4,10,9,10,9,6,5,12,11,16,15,3,2,8,7,14,13]
        for jIdx in range(len(mapping)):
            shelf17[:,jIdx,:] = keypoints2d[:,mapping[jIdx],:]
        return shelf17

    def associate_frame(
            self, kps2d_paf:list
    ) -> Tuple[Keypoints, List[int]]:
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
        self.n_kps = len(kps2d_paf[0]['joints'])
        m_personsMap = self.fourd_matching(kps2d_paf, self.last_multi_kps3d)
        m_skels2d = self.cal_keypoints2d(m_personsMap, kps2d_paf)
        ####
        multi_kps3d = []
        identities = []
        for person_id in m_skels2d:
            #convert
            keypoints2d = np.zeros((self.n_views, self.n_kps, 3))
            for view in range(self.n_views):
                for joint_id in range(self.n_kps):
                    keypoints2d[view][joint_id] = m_skels2d[person_id][:,view*self.n_kps+joint_id]
            # keypoints2d = self.MappingToSkel17(m_skels2d[person_id])
            matched_mkps2d = np.zeros((self.n_views, self.n_kps, 2))
            matched_mkps2d_mask = np.zeros((self.n_views, self.n_kps, 1))
            matched_mkps2d_conf = np.zeros((self.n_views, self.n_kps, 1))
            matched_mkps2d = keypoints2d[...,:2] 
            matched_mkps2d_mask = np.ones_like(keypoints2d[..., 0:1])
            matched_mkps2d_conf[...,0] = keypoints2d[...,2]
            selected_mask = self.point_selector.get_selection_mask(
                    np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                   axis=-1), matched_mkps2d_mask)
            kps3d = self.triangulator.triangulate(matched_mkps2d,
                                                      selected_mask)            

            if not np.isnan(kps3d).all():
                multi_kps3d.append(kps3d)
                # identities.append(person_id)
        multi_kps3d = np.array(multi_kps3d)
        
        if len(multi_kps3d) > 0:
            keypoints3d, identities = self.assign_identities_frame(multi_kps3d)
            self.last_multi_kps3d = dict()
            # for pid in identities:
            #     self.last_multi_kps3d[pid] = keypoints3d.get_keypoints()[0, pid,...].T
        else:
            keypoints3d = Keypoints()
            identities = []
            self.last_multi_kps3d = dict()
        # ####

        # ####
        # multi_kps3d = self.triangulator.triangulate(m_skels2d)
        # self.last_multi_kps3d = multi_kps3d
        # kps_arr = np.zeros((1,len(multi_kps3d),self.n_kps,4))
        # mask_arr = np.zeros((1,len(multi_kps3d),self.n_kps))
        # for index, pid in enumerate(multi_kps3d):            
        #     kps_arr[0,index,...] = multi_kps3d[pid][:,:self.n_kps].T
        #     mask_arr[0,index,:] = multi_kps3d[pid][3,:self.n_kps]
        # keypoints3d = Keypoints(kps=kps_arr, mask=mask_arr, convention=self.kps_convention)
        # identities = multi_kps3d.keys()
        ###
        
        return keypoints3d, identities

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

    
