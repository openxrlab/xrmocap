# yapf: disable
import logging
import numpy as np
from typing import List, Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_metric import BaseMetric

# yapf: enable


class PCKMetric(BaseMetric):
    """PCK metric measures accuracy of the localization of the body joints.

    This is a rank-2 metric, it depends on rank-1 metric pa_mpjpe or mpjpe.
    """
    RANK = 2

    def __init__(
        self,
        name: str,
        threshold: Union[List[int], List[float]] = [50, 100],
        use_pa_mpjpe: bool = False,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init PCK metric evaluation.

        Args:
            name (str):
                Name of the metric.
            threshold (Union[List[int],List[float]]):
                A list of threshold for PCK evaluation.
            use_pa_mpjpe (bool, optional):
                Whether to use PA-MPJPE instead of MPJPE.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """
        BaseMetric.__init__(self, name=name, logger=logger)
        self.threshold = threshold
        self.use_pa_mpjpe = use_pa_mpjpe

    def __call__(self, pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                 **kwargs):
        if self.use_pa_mpjpe and 'pa_mpjpe_value' in kwargs:
            raw_mpjpe_value = kwargs['pa_mpjpe_value']
        elif not self.use_pa_mpjpe and 'mpjpe_value' in kwargs:
            raw_mpjpe_value = kwargs['mpjpe_value']
        else:
            self.logger.error('No mpjpe metric found. '
                              'Please add MPJPEMetric or PAMPJPEMetric '
                              'in the config.')
            raise KeyError

        gt_mask = gt_keypoints3d.get_mask()
        masked_mpjpe_value = raw_mpjpe_value[np.where(gt_mask > 0)]

        pck_value = {}
        for thr in self.threshold:
            pck_thr = np.mean(masked_mpjpe_value <= thr) * 100
            pck_value[f'pck@{str(thr)}'] = pck_thr

        return pck_value
