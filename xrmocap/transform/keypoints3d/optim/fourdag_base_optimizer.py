import numpy as np

from xrmocap.utils.fourdag_utils import skel_info
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from typing import Union


class FourDAGBaseOptimizer():

    def __init__(self,
                 triangulator: Union[None, dict, BaseTriangulator] = None,
                 kps_convention='fourdag_19',
                 min_triangulate_cnt: int = 15,
                 triangulate_thresh: float = 0.05,
                 logger=None):

        if isinstance(triangulator, dict):
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        self.kps_convention = kps_convention
        self.min_triangulate_cnt = min_triangulate_cnt
        self.triangulate_thresh = triangulate_thresh

        self.projs = None
        self.m_skels = dict()
        self.m_skelInfos = []
    
    def triangulate_person(self, skel2d):
        points2d = skel2d.T.reshape((-1, skel_info[self.kps_convention]['n_kps'], 3))
        points3d = self.triangulator.triangulate(points2d)
        mask = self.triangulator.loss < self.triangulate_thresh
        skel = np.concatenate((points3d.T, mask.reshape(1,-1)), axis=0)
        return skel

    def set_cameras(self, cameras):
        self.triangulator.set_cameras(cameras)

    def update(self, skels2d):
        skelIter = iter(self.m_skels.copy())
        prevCnt = len(self.m_skels)
        for pIdx, corr_id in enumerate(skels2d):
            skel = self.triangulate_person(skels2d[corr_id])
            active = sum(skel[3] > 0) >= self.min_triangulate_cnt
            if pIdx < prevCnt:
                if active:
                    self.m_skels[next(skelIter)] = skel
                else:
                    self.m_skels.pop(next(skelIter))

            elif active:
                self.m_skels[corr_id] = skel

        return self.m_skels
