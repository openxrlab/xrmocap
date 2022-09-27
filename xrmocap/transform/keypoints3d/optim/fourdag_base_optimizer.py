import numpy as np

from xrmocap.utils.fourdag_utils import LIMB_INFO
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
        """Base class for fourdag optimizater.

        Args:
            triangulator: 
                triangulator to construct 3D keypoints  
            kps_convention (str):
                The name of keypoints convention.
            min_triangulate_cnt (int):
                the minimum amount of 3D keypoints to be accepted
            triangulate_thresh (float):
                the maximal triangulate loss to be accepted
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        if isinstance(triangulator, dict):
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        self.kps_convention = kps_convention
        self.min_triangulate_cnt = min_triangulate_cnt
        self.triangulate_thresh = triangulate_thresh

        self.projs = None
        self.trace_limbs = dict()
        self.trace_limb_infos = dict()
    
    def triangulate_person(self, limb2d):
        points2d = limb2d.T.reshape((-1, LIMB_INFO[self.kps_convention]['n_kps'], 3))
        points3d = self.triangulator.triangulate(points2d)
        mask = self.triangulator.loss < self.triangulate_thresh
        limb = np.concatenate((points3d.T, mask.reshape(1,-1)), axis=0)
        return limb

    def set_cameras(self, camera_parameters):
        projs = np.zeros((3, len(camera_parameters) * 4))
        for view in range(len(camera_parameters)):
            K = camera_parameters[view].intrinsic33()
            T = np.array(camera_parameters[view].get_extrinsic_t())
            R = np.array(camera_parameters[view].get_extrinsic_r())
            Proj = np.zeros((3, 4), dtype=np.float)
            for i in range(3):
                for j in range(4):
                    Proj[i, j] = R[i, j] if j < 3 else T[i]
            projs[:, 4 * view:4 * view + 4] = np.matmul(K, Proj)
        self.triangulator.set_cameras(camera_parameters)
        self.triangulator.set_proj_mat(projs)
        self.projs = projs

    def update(self, limbs2d):
        for pidx, corr_id in enumerate(limbs2d):
            limb = self.triangulate_person(limbs2d[corr_id])
            active = sum(limb[3] > 0) >= self.min_triangulate_cnt
            if corr_id in self.trace_limbs:
                if active:
                    self.trace_limbs[corr_id] = limb
                else:
                    self.trace_limbs.pop(corr_id)
            elif active:
                self.trace_limbs[corr_id] = limb

        return self.trace_limbs
