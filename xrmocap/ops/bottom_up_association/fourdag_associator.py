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
# yapf: enable

class FourDAGAssociator:

    def __init__(self,
                 triangulator: Union[None, dict, BaseTriangulator],
                 fourd_matching: Union[None, dict] = None,
                 min_asgn_cnt: int = 5 ,
                 logger: Union[None, str, logging.Logger] = None) -> None:

        self.logger = get_logger(logger)

        if isinstance(triangulator, dict):
            triangulator['logger'] = self.logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator
        
        self.n_views = -1
        self.last_multi_kps3d = dict()
        self.n_kps = 19
        self.min_asgn_cnt = min_asgn_cnt
        if isinstance(fourd_matching, dict):
            fourd_matching['logger'] = self.logger
            self.fourd_matching = build_matching(fourd_matching)
        else:
            self.fourd_matching = fourd_matching

    def set_cameras(
        self, cameras: List[Union[FisheyeCameraParameter,
                                  PinholeCameraParameter]]
    ) -> None:
        
        self.triangulator.set_cameras(cameras)
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

    def MappingToCampus(self,multi_kps3d):
        kps_arr = np.zeros((1,len(multi_kps3d),14,4))
        mask_arr = np.zeros((1,len(multi_kps3d),14))
        for index, pid in enumerate(multi_kps3d):
            fourd19 = multi_kps3d[pid]
            fourd15 = np.zeros((4,15), dtype=np.float)
            mapping = [ 13, 7, 2, 3, 8, 14, 15, 11, 5, 6, 12, 16, 1, 4, 0 ]
            for jIdx in range(len(mapping)):
                fourd15[:,jIdx] = fourd19[:,mapping[jIdx]]

            faceDir = np.cross((fourd15[0:0+3,12:12+1] - fourd15[0:0+3,14:14+1]).reshape(-1),(fourd15[:3,8:9] - fourd15[:3,9:10]).reshape(-1))
            faceDir = faceDir / np.linalg.norm(faceDir)
            zDir = np.array([0., 0., 1.], dtype=np.float)
            shoulderCenter = (fourd19[:3,5:6] + fourd19[:3,6:7]) / 2.
            headCenter = (fourd19[:3,9:10] + fourd19[:3,10:11]) / 2.
            
            fourd15[:3,12:13] = shoulderCenter + (headCenter - shoulderCenter)*0.5
            if not (fourd15[3,12] == 0 or  fourd15[3,14] == 0 or  fourd15[3,8] == 0 or  fourd15[3,9] == 0):
                fourd15[:3,13:14] = fourd15[:3,12:13] + faceDir.reshape(-1,1) * 0.125 + zDir.reshape(-1,1) * 0.145

            kps_arr[0,index,...] = fourd15[:,:14].T
            mask_arr[0,index,:] = fourd15[3,:14]

        keypoints = Keypoints(kps=kps_arr, mask=mask_arr, convention='campus')
        return keypoints

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
           
        m_personsMap = self.fourd_matching(kps2d_paf, self.last_multi_kps3d)
        m_skels2d = self.cal_keypoints2d(m_personsMap, kps2d_paf)
        multi_kps3d = self.triangulator.triangulate(m_skels2d)
        self.last_multi_kps3d = multi_kps3d
        keypoints3d = self.MappingToCampus(multi_kps3d) 
        indentities = multi_kps3d.keys()

        return keypoints3d, indentities

    
