# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.triangulation.builder import (
    AniposelibTriangulator, build_triangulator,
)
from xrmocap.transform.limbs import get_limbs_from_keypoints
from .base_optimizer import BaseOptimizer

# yapf: enable


class AniposelibOptimizer(BaseOptimizer):

    def __init__(self,
                 triangulator: Union[AniposelibTriangulator, dict],
                 smooth_weight: float = 4.0,
                 bone_length_weight: float = 2.0,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Call optim_points in aniposelib to apply smooth and constraint on a
        keypoints3d instance.

        Args:
            triangulator (Union[AniposelibTriangulator, dict]):
                Triangulator or config dict.
            smooth_weight (float, optional):
                Weight of smooth optimization.
                Defaults to 4.0.
            bone_length_weight (float, optional):
                Weight of bone length optimization.
                Defaults to 2.0.
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        self.smooth_weight = smooth_weight
        self.bone_length_weight = bone_length_weight
        if isinstance(triangulator, dict):
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             mview_kps2d: np.ndarray,
                             mview_kps2d_mask: np.ndarray,
                             **kwargs: dict) -> Keypoints:
        """Forward function of keypoints3d optimizer.

        Args:
            keypoints3d (Keypoints): Input keypoints3d.
            mview_kps2d (np.ndarray):
                Multi-view keypoints2d array for triangulation,
                in shape [n_view, n_frame, n_person, n_kps, 2+n],
                n >= 0.
            mview_kps2d_mask (np.ndarray):
                Multi-view keypoints2d mask for triangulation,
                in shape [n_view, n_frame, n_person, n_kps, 1].

        kwargs:
            Redundant keyword arguments to be
            ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        if keypoints3d.dtype == 'numpy':
            keypoints3d_np = keypoints3d
        else:
            keypoints3d_np = keypoints3d.to_numpy()
            self.logger.warning('AniposelibOptimizer only support numpy kps,' +
                                ' the input kps has been converted to numpy.')
        limbs = get_limbs_from_keypoints(keypoints3d)
        n_person = keypoints3d.get_person_number()
        kps3d_src = keypoints3d.get_keypoints()
        kps3d_dst = kps3d_src.copy()
        ignore_idxs = np.where(mview_kps2d_mask != 1)
        mview_kps2d[ignore_idxs[0], ignore_idxs[1], ignore_idxs[2],
                    ignore_idxs[3], :] = np.nan
        connections = limbs.get_connections()
        camera_group = self.triangulator.__prepare_aniposelib_camera__()
        for person_idx in range(n_person):
            kps2d = mview_kps2d[:, :, person_idx, :, :2]
            kps3d = kps3d_src[:, person_idx, :, :3]
            kps3d_dst[:, person_idx, :, :3] = camera_group.optim_points(
                kps2d,
                kps3d,
                scale_smooth=self.smooth_weight,
                scale_length=self.bone_length_weight,
                constraints=connections,
                verbose=self.verbose,
            )
        ret_keypoints3d = keypoints3d_np.clone()
        ret_keypoints3d.set_keypoints(kps3d_dst)
        return ret_keypoints3d
