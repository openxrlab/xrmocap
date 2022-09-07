# yapf: disable
import logging
import numpy as np
from typing import List, Tuple, Union
import json
import cv2
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints

from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)

from xrmocap.ops.fourd_association.matching.builder import build_matching
# yapf: enable


class Camera():
    def __init__(self,cam_param) -> None:
        super().__init__()
        self.originK = np.zeros((3,3), dtype=np.float)
        self.cvR = np.zeros((3,3), dtype=np.float)
        self.cvT = np.zeros(3, dtype=np.float)

        self.originK = cam_param.intrinsic33()
        self.cvT = np.array(cam_param.get_extrinsic_t())
        self.cvR = np.array(cam_param.get_extrinsic_r())
        self.imgSize = [cam_param.width, cam_param.height]
        self.rectifyAlpha = 0
        # self.cvK, _ = cv2.getOptimalNewCameraMatrix(self.originK,self.distCoeff, self.imgSize, self.rectifyAlpha)
        self.cvK = self.originK

        self.cvKi = np.linalg.inv(self.cvK)
        self.cvRt = self.cvR.T
        self.cvRtKi = np.matmul(self.cvRt, self.cvKi)
        self.cvPos = -np.matmul(self.cvRt, self.cvT) 
        
        self.cvProj = np.zeros((3,4), dtype=np.float)
        for i in range(3):
            for j in range(4):
                self.cvProj[i,j] = self.cvR[i,j] if j < 3 else self.cvT[i]
        self.cvProj = np.matmul(self.cvK, self.cvProj)

    def calcRay(self, uv):
        ver  = -self.cvRtKi.dot(np.append(uv, 1).T)
        return ver / np.linalg.norm(ver)
        
def convert_kps3d(multi_kps3d):
    kps_arr = np.zeros((1,len(multi_kps3d),14,4))
    for index, skel15 in enumerate(multi_kps3d):
        kps_arr[0,index,...] = skel15.T
    mask_arr = np.zeros((1,len(multi_kps3d),14))
    for index, skel15 in enumerate(multi_kps3d):
        mask_arr[0,index,:] = skel15[3,:]
    keypoints = Keypoints(kps=kps_arr, mask=mask_arr, convention='campus')
    return keypoints

class FourdAssociator:

    def __init__(self,
                 triangulator: Union[None, dict, BaseTriangulator],
                 m_filter: bool=False,
                 fourd_matching: Union[None, dict] = None,
                 m_minAsgnCnt: int = 5 ,
                 logger: Union[None, str, logging.Logger] = None) -> None:

        self.logger = get_logger(logger)

        if isinstance(triangulator, dict):
            triangulator['logger'] = self.logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator
        
        self.n_views = -1
        self.last_multi_kps3d = dict()
        self.joint_size = 19
        self.m_minAsgnCnt = m_minAsgnCnt
        self.m_filter = m_filter
        if isinstance(fourd_matching, dict):
            fourd_matching['logger'] = self.logger
            self.fourd_matching = build_matching(fourd_matching)
        else:
            self.fourd_matching = fourd_matching

    def set_cameras(
        self, cameras: List[Union[FisheyeCameraParameter,
                                  PinholeCameraParameter]]
    ) -> None:
        ###tricky
        mycameras = []
        for view in range(len(cameras)):
            mycameras.append(Camera(cameras[view]))
        
        self.triangulator.set_cameras(mycameras)
        self.fourd_matching.set_cameras(mycameras)
        self.n_views = len(cameras)


    def cal_keypoints2d(self, m_personsMap, kps2d_paf):
    
        for i, person_id in enumerate(m_personsMap.copy()):
            if i < len(self.last_multi_kps3d):
                continue
            if  sum(sum(m_personsMap[person_id] >= 0)) >= self.m_minAsgnCnt:
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
            skel2d = np.zeros((3,self.n_views *self.joint_size))
            for view in range(self.n_views):
                for joint_id in range(self.joint_size):
                    index = m_personsMap[person_id][joint_id, view]
                    if index != -1:
                        skel2d[:,view*self.joint_size+joint_id] = kps2d_paf[view]['joints'][joint_id][index]
                    else:
                        continue

            m_skels2d[identity] = skel2d
        return m_skels2d

    def MappingToCampus(self,skel19):
        shelf15 = np.zeros((4,15), dtype=np.float)
        mapping = [ 13, 7, 2, 3, 8, 14, 15, 11, 5, 6, 12, 16, 1, 4, 0 ]
        for jIdx in range(len(mapping)):
            shelf15[:,jIdx] = skel19[:,mapping[jIdx]]

        faceDir = np.cross((shelf15[0:0+3,12:12+1] - shelf15[0:0+3,14:14+1]).reshape(-1),(shelf15[:3,8:9] - shelf15[:3,9:10]).reshape(-1))
        faceDir = faceDir / np.linalg.norm(faceDir)
        zDir = np.array([0., 0., 1.], dtype=np.float)
        shoulderCenter = (skel19[:3,5:6] + skel19[:3,6:7]) / 2.
        headCenter = (skel19[:3,9:10] + skel19[:3,10:11]) / 2.
        
        shelf15[:3,12:13] = shoulderCenter + (headCenter - shoulderCenter)*0.5
        if shelf15[3,12] == 0 or  shelf15[3,14] == 0 or  shelf15[3,8] == 0 or  shelf15[3,9] == 0:
            return shelf15[:,:14]
        shelf15[:3,13:14] = shelf15[:3,12:13] + faceDir.reshape(-1,1) * 0.125 + zDir.reshape(-1,1) * 0.145
        return shelf15[:,:14]

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
        if self.m_filter:
            multi_kps3d = self.triangulator.triangulate_w_filter(m_skels2d)
        else:
            multi_kps3d = self.triangulator.triangulate_wo_filter(m_skels2d)
        self.last_multi_kps3d = multi_kps3d

        shelfSkels = []
        for skel_id in multi_kps3d:
            shelfSkels.append(self.MappingToCampus(multi_kps3d[skel_id]))
            
        keypoints3d = convert_kps3d(shelfSkels)
        indentities = multi_kps3d.keys()
        return keypoints3d, indentities

    
