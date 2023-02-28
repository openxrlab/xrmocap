# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.eval_utils import align_by_keypoint
from .base_metric import BaseMetric

# yapf: enable


class MPJPEMetric(BaseMetric):
    """Mean per-joint position error(MPJPE).

    This is a rank-1 metric, depends on rank-0 metric PredictionMatcher. If the
    number of prediction does not align with the number of ground truth, this
    metric will evaluate predictions matched to the ground truth.
    """
    RANK = 1

    def __init__(
        self,
        name: str,
        align_kps_name: Union[None, str] = None,
        unit_scale: Union[None, int] = None,
        outlier_threshold: Union[float, int, None] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init MPJPE metric evaluation.

        Args:
            name (str):
                Name of the metric.
            align_kps_name (Union[None, str], optional):
                Whether to align keypoints3d with a given keypoint.
                Defaults to None.
            unit_scale (Union[None,int], optional):
                Scale factor to convert prediction and ground truth value.
                Make sure the unit of prediction and ground truth is align
                with the thresholds. Defaults to None.
            outlier_threshold (Union[float, int, None], optional):
                Threshold to remove outliers in prediction. Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """
        BaseMetric.__init__(self, name=name, logger=logger)
        self.align_kps_name = align_kps_name
        self.unit_scale = unit_scale
        self.outlier_threshold = outlier_threshold

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

        gt_mask = gt_keypoints3d.get_mask()
        gt_n_frame = gt_keypoints3d.get_keypoints().shape[0]
        pred_n_frame = pred_keypoints3d.get_keypoints().shape[0]
        if gt_n_frame == pred_n_frame:
            n_frame = gt_n_frame
        else:
            self.logger.error('Prediction and ground-truth does not match in '
                              'the number of frame.')
            raise ValueError

        if 'match_matrix_gt2pred' and 'match_matrix_pred2gt' in kwargs:
            match_matrix_gt2pred = kwargs['match_matrix_gt2pred']
            match_matrix_pred2gt = kwargs['match_matrix_pred2gt']
        else:
            self.logger.error('No matching matrix found. '
                              'Please add PredictionMatcher in the config.')
            raise KeyError

        if self.align_kps_name is None:
            pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
            gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        else:
            pred_kps3d = align_by_keypoint(pred_keypoints3d,
                                           self.align_kps_name)[..., :3]
            gt_kps3d = align_by_keypoint(gt_keypoints3d,
                                         self.align_kps_name)[..., :3]

        sort_pred_kps3d = np.zeros_like(gt_kps3d)
        sort_gt_kps3d = np.zeros_like(pred_kps3d)
        for frame_idx in range(n_frame):
            match_list = match_matrix_gt2pred[frame_idx]
            sort_pred_kps3d[frame_idx, :] = pred_kps3d[frame_idx, match_list]

            match_list = match_matrix_pred2gt[frame_idx]
            sort_gt_kps3d[frame_idx, :] = gt_kps3d[frame_idx, match_list]

        mpjpe_value_gt2pred = np.linalg.norm(
            gt_kps3d - sort_pred_kps3d, ord=2, axis=-1)
        mpjpe_value_pred2gt = np.linalg.norm(
            sort_gt_kps3d - pred_kps3d, ord=2, axis=-1)

        if self.unit_scale is not None:
            mpjpe_value_gt2pred = mpjpe_value_gt2pred * self.unit_scale
            mpjpe_value_pred2gt = mpjpe_value_pred2gt * self.unit_scale

        masked_mpjpe_value = mpjpe_value_gt2pred[np.where(gt_mask > 0)]
        if self.outlier_threshold is not None:
            masked_mpjpe_value = masked_mpjpe_value[np.where(
                masked_mpjpe_value < self.outlier_threshold)]
        mpjpe_mean, mpjpe_std = np.mean(masked_mpjpe_value), np.std(
            masked_mpjpe_value)

        return dict(
            mpjpe_mean=mpjpe_mean,
            mpjpe_std=mpjpe_std,
            mpjpe_value=mpjpe_value_gt2pred,
            mpjpe_value_pred2gt=mpjpe_value_pred2gt,
        )
