# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.eval_utils import align_by_keypoint
from xrmocap.utils.mvpose_utils import vectorize_distance
from .base_metric import BaseMetric

# yapf: enable


class PredictionMatcher(BaseMetric):
    """Prediction matcher matches the predictions with ground truth when the
    number of prediction does not align with the number of ground truth, this
    metric will return pred2gt match and gt2pred matching for various metric
    usage.

    This is a rank-0 module.
    """
    RANK = 0

    def __init__(
        self,
        name: str,
        align_kps_name: Union[None, str] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init PredictionMatcher.

        Args:
            name (str):
                Name of the metric.
            align_kps_name (Union[None, str], optional):
                Whether to align keypoints3d with a given keypoint.
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """
        BaseMetric.__init__(self, name=name, logger=logger)
        self.align_kps_name = align_kps_name

    def __call__(self, pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                 **kwargs) -> dict:

        pred_kps3d_convention = pred_keypoints3d.get_convention()
        gt_kps3d_convention = gt_keypoints3d.get_convention()
        if pred_kps3d_convention != gt_kps3d_convention:
            self.logger.error('Predicted keypoints3d and gt keypoints3d '
                              'is having different convention.')
            raise ValueError
        else:
            self.convention = gt_kps3d_convention

        gt_n_frame, gt_n_person = gt_keypoints3d.get_keypoints().shape[:2]
        pred_n_frame, pred_n_person = pred_keypoints3d.get_keypoints(
        ).shape[:2]
        if gt_n_frame == pred_n_frame:
            n_frame = gt_n_frame
        else:
            self.logger.error('Prediction and ground-truth does not match in '
                              'the number of frame')
            raise ValueError

        if self.align_kps_name is None:
            pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
            gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        else:
            pred_kps3d = align_by_keypoint(pred_keypoints3d,
                                           self.align_kps_name)[..., :3]
            gt_kps3d = align_by_keypoint(gt_keypoints3d,
                                         self.align_kps_name)[..., :3]

        # get distance matrix between gt and pred
        # (n_frame, gt_n_person, pred_n_person)
        dist = np.zeros((n_frame, gt_n_person, pred_n_person))
        for frame_idx in range(n_frame):
            dist[frame_idx, :] = vectorize_distance(gt_kps3d[frame_idx, :],
                                                    pred_kps3d[frame_idx, :])

        # find the best pred for each gt given
        # filter the matching for invalid gt with gt mask
        # -1 for invalid gt, pred_index for valid gt
        match_matrix_gt2pred = np.argmin(dist, axis=2)
        for frame_idx in range(n_frame):
            gt_frame_mask = gt_keypoints3d.get_mask()[frame_idx]
            invalid_gt = np.where(np.sum(gt_frame_mask, axis=1) == 0)
            match_matrix_gt2pred[frame_idx, invalid_gt] = -1

        # find the best gt for each pred given
        match_matrix_pred2gt = np.argmin(dist, axis=1)

        return dict(
            match_matrix_gt2pred=match_matrix_gt2pred,
            match_matrix_pred2gt=match_matrix_pred2gt,
        )
