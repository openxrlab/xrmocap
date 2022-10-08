# yapf: disable
import numpy as np
from typing import Union

from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.utils.fourdag_utils import LimbInfo

# yapf: enable


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
        self.limb_info = LimbInfo(self.kps_convention)

    def triangulate_person(self, limb2d):
        kps2d = limb2d.T.reshape((-1, self.limb_info.get_kps_number(), 3))
        kps3d = self.triangulator.triangulate(kps2d)
        mask = self.triangulator.loss < self.triangulate_thresh
        limb = np.concatenate((kps3d.T, mask.reshape(1, -1)), axis=0)
        return limb

    def set_cameras(self, camera_parameters):
        self.triangulator.set_cameras(camera_parameters)
        self.projs = self.triangulator.projs

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
