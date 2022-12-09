# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.eval_utils import align_by_keypoint
from .base_metric import BaseMetric

# yapf: enable


class MPJPEMetric(BaseMetric):
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
        if self.align_kps_name is None:
            pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
            gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        else:
            pred_kps3d = align_by_keypoint(pred_keypoints3d,
                                           self.align_kps_name)
            gt_kps3d = align_by_keypoint(gt_keypoints3d, self.align_kps_name)
        mpjpe_value = np.sqrt(
            np.sum(np.square(pred_kps3d - gt_kps3d), axis=-1))
        return dict(mpjpe_value=mpjpe_value)
