import numpy as np
from xrmocap.utils.fourdag_utils import *

class Triangulator():
    def __init__(self):
        self.points = None
        self.projs = None
        self.convergent = False
        self.loss = 0
        self.pos = np.zeros(3, dtype=np.float32)

    def solve(self,maxIterTime = 20, updateTolerance = 1e-4, regularTerm = 1e-4):
        self.convergent = False
        self.loss = 10e9
        self.pos = np.zeros(3, dtype=np.float32)

        if sum(self.points[2] > 0) < 2:
            return

        for iterTime in range(maxIterTime):
            if self.convergent:
                break
            ATA = regularTerm * np.identity(3,dtype=np.float32)
            ATb = np.zeros(3,dtype=np.float32)
            for view in range(self.points.shape[1]):
                if self.points[2, view] > 0: 
                    proj = self.projs[:,4 * view: 4 * view+ 4]
                    xyz = np.matmul(proj, np.append(self.pos, 1))
                    jacobi = np.zeros((2, 3),dtype=np.float32)
                    jacobi = np.array([1.0 / xyz[2], 0.0, -xyz[0] / (xyz[2]*xyz[2]),
                        0.0, 1.0 / xyz[2], -xyz[1] / (xyz[2]*xyz[2])],dtype=np.float32).reshape((2,3))
                    jacobi = np.matmul(jacobi, proj[:,:3])
                    w = self.points[2, view]
                    ATA += w *np.matmul(jacobi.T, jacobi)
                    ATb += w * np.matmul(jacobi.T,(self.points[:,view][:2] - xyz[:2]/xyz[2]))

            delta = np.linalg.solve(ATA, ATb)
            self.loss = np.linalg.norm(delta)
            if np.linalg.norm(delta) < updateTolerance:
                self.convergent = True
            else:
                self.pos += delta

class BaseOptimization():
    def __init__(self,
                 kps_convention = 'fourdag_19',
                 min_triangulate_cnt: int=15,
                 triangulate_thresh: float=0.05,
                 logger=None):

        self.kps_convention = kps_convention
        self.min_triangulate_cnt = min_triangulate_cnt
        self.triangulate_thresh = triangulate_thresh

        self.projs = None
        self.m_skels = dict()
        self.m_skelInfos = []

    def triangulate_person(self, skel2d):
        skel = np.zeros((4, skel_info[self.kps_convention]['n_kps']),dtype=np.float32)
        triangulator = Triangulator()
        triangulator.projs = self.projs
        triangulator.points = np.zeros((3,int(self.projs.shape[1]/4)),dtype=np.float32)
        for jIdx in range(skel_info[self.kps_convention]['n_kps']):
            for view in range(int(self.projs.shape[1] / 4)):
                skel_tmp = skel2d[:,view*skel_info[self.kps_convention]['n_kps']:view*skel_info[self.kps_convention]['n_kps']+skel_info[self.kps_convention]['n_kps']]
                triangulator.points[:,view] = skel_tmp[:,jIdx]
            triangulator.solve()
            if triangulator.loss < self.triangulate_thresh:
                skel[:,jIdx] = np.append(triangulator.pos, 1)
        return skel


    def set_cameras(self, cameras):
        self.projs = np.zeros((3, len(cameras) * 4))
        for view in range(len(cameras)):
            K = cameras[view].intrinsic33()
            T = np.array(cameras[view].get_extrinsic_t())
            R = np.array(cameras[view].get_extrinsic_r())        
            Proj = np.zeros((3,4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    Proj[i,j] = R[i,j] if j < 3 else T[i]
            self.projs[:,4*view:4*view + 4] = np.matmul(K, Proj)

    def update(self, skels2d):
        skelIter = iter(self.m_skels.copy())
        prevCnt = len(self.m_skels)
        for pIdx, corr_id in enumerate(skels2d):
            skel = self.triangulate_person(skels2d[corr_id])
            active = sum(skel[3]> 0) >= self.min_triangulate_cnt
            if pIdx < prevCnt:
                if active:
                    self.m_skels[next(skelIter)] = skel
                else:
                    self.m_skels.pop(next(skelIter))

            elif active:
                self.m_skels[corr_id] = skel

        return self.m_skels