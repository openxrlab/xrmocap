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
            triangulator['logger'] = self.logger
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
        skel = np.zeros((4, skel_info[self.kps_convention]['n_kps']),
                        dtype=np.float32)
        self.triangulator.projs = self.projs
        self.triangulator.points = np.zeros((3, int(self.projs.shape[1] / 4)),
                                       dtype=np.float32)
        for jIdx in range(skel_info[self.kps_convention]['n_kps']):
            for view in range(int(self.projs.shape[1] / 4)):
                skel_tmp = skel2d[:, view *
                                  skel_info[self.
                                            kps_convention]['n_kps']:view *
                                  skel_info[self.kps_convention]['n_kps'] +
                                  skel_info[self.kps_convention]['n_kps']]
                self.triangulator.points[:, view] = skel_tmp[:, jIdx]
            self.triangulator.solve()
            if self.triangulator.loss < self.triangulate_thresh:
                skel[:, jIdx] = np.append(self.triangulator.pos, 1)
        return skel

    def set_cameras(self, cameras):
        self.projs = np.zeros((3, len(cameras) * 4))
        for view in range(len(cameras)):
            K = cameras[view].intrinsic33()
            T = np.array(cameras[view].get_extrinsic_t())
            R = np.array(cameras[view].get_extrinsic_r())
            Proj = np.zeros((3, 4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    Proj[i, j] = R[i, j] if j < 3 else T[i]
            self.projs[:, 4 * view:4 * view + 4] = np.matmul(K, Proj)

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
