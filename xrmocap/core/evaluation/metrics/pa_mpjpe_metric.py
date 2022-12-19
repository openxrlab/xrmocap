# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.eval_utils import (
    align_by_keypoint, compute_similarity_transform,
)
from .base_metric import BaseMetric

# yapf: enable


class PAMPJPEMetric(BaseMetric):
    RANK = 1

    def __init__(
        self,
        name: str,
        align_kps_name: Union[None, str] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        BaseMetric.__init__(self, name=name, logger=logger)
        self.align_kps_name = align_kps_name

    def __call__(self, pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                 **kwargs):
        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        _, _, rotation, scaling, transl = compute_similarity_transform(
            gt_kps3d.reshape(-1, 3),
            pred_kps3d.reshape(-1, 3),
            compute_optimal_scale=True)
        pred_kps3d_pa = (scaling * pred_kps3d.dot(rotation)) + transl

        pred_keypoints3d_pa = pred_keypoints3d.clone()
        pred_kps3d_conf = pred_keypoints3d.get_keypoints()[..., -1:]
        pred_keypoints3d_pa.set_keypoints(
            np.concatenate((pred_kps3d_pa, pred_kps3d_conf), axis=-1))
        if self.align_kps_name is None:
            pred_kps3d_pa = pred_keypoints3d_pa.get_keypoints()[..., :3]
            gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        else:
            pred_kps3d_pa = align_by_keypoint(pred_keypoints3d_pa,
                                              self.align_kps_name)
            gt_kps3d = align_by_keypoint(gt_keypoints3d, self.align_kps_name)
        pa_mpjpe_value = np.sqrt(
            np.sum(np.square(pred_kps3d_pa - gt_kps3d), axis=-1))
        return dict(pa_mpjpe_value=pa_mpjpe_value)
