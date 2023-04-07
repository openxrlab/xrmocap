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
    """MPJPE after further alignment(Procrustes analysis (PA)).

    This is a rank-1 metric, depends on rank-0 metric PredictionMatcher. If the
    number of prediction does not align with the number of ground truth, this
    metric will evaluate predictions matched to the ground truth.
    """
    RANK = 1

    def __init__(
        self,
        name: str,
        unit_scale: Union[None, int] = None,
        align_kps_name: Union[None, str] = None,
        outlier_threshold: Union[float, int, None] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init PA-MPJPE metric evaluation.

        Args:
            name (str):
                Name of the metric.
            align_kps_name (Union[None, str], optional):
                Whether to align keypoints3d with a given keypoint.
                Align by keypoint is not recommended for PA-MPJPE metric.
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

        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]

        gt_mask = gt_keypoints3d.get_mask()
        gt_n_frame, gt_n_person = gt_kps3d.shape[:2]
        pred_n_frame, pred_n_person = pred_kps3d.shape[:2]

        if gt_n_frame == pred_n_frame:
            n_frame = gt_n_frame
        else:
            self.logger.error('Prediction and ground-truth does not match in '
                              'the number of frame')
            raise ValueError

        if 'match_matrix_gt2pred' in kwargs:
            match_matrix_gt2pred = kwargs['match_matrix_gt2pred']
        else:
            self.logger.error('No matching metric found. '
                              'Please add PredictionMatcher in the config.')
            raise KeyError

        sort_pred_kps3d = np.zeros_like(gt_kps3d)
        sort_pred_kps3d_pa = np.zeros_like(gt_kps3d)
        for frame_idx in range(n_frame):
            match_list = match_matrix_gt2pred[frame_idx]
            sort_pred_kps3d[frame_idx, :] = pred_kps3d[frame_idx, match_list]

        for frame_idx in range(n_frame):
            for person_idx in range(gt_n_person):
                person_mask = gt_mask[frame_idx, person_idx, :]
                masked_gt_kps3d = gt_kps3d[frame_idx, person_idx,
                                           np.where(
                                               person_mask > 0), :].reshape(
                                                   -1, 3)
                masked_sorted_pred_kps3d = sort_pred_kps3d[
                    frame_idx, person_idx,
                    np.where(person_mask > 0), :].reshape(-1, 3)
                if np.all((masked_sorted_pred_kps3d
                           == 0)) or len(masked_sorted_pred_kps3d) == 0:
                    continue
                _, _, rotation, scaling, transl = compute_similarity_transform(
                    masked_gt_kps3d,
                    masked_sorted_pred_kps3d,
                    compute_optimal_scale=True)
                pred_kps3d_pa = (scaling * sort_pred_kps3d[
                    frame_idx, person_idx].dot(rotation)) + transl

                sort_pred_kps3d_pa[frame_idx, person_idx, :] = pred_kps3d_pa

        if self.align_kps_name:
            sort_pred_keypoints3d_pa = Keypoints(
                dtype='numpy',
                kps=sort_pred_kps3d_pa,
                mask=gt_mask,
                convention=gt_keypoints3d.get_convention(),
                logger=self.logger)
            sort_pred_kps3d_pa = align_by_keypoint(
                sort_pred_keypoints3d_pa, self.align_kps_name)[..., :3]
            gt_kps3d = align_by_keypoint(gt_keypoints3d,
                                         self.align_kps_name)[..., :3]

        pa_mpjpe_value = np.linalg.norm(
            gt_kps3d - sort_pred_kps3d_pa, ord=2, axis=-1)

        if self.unit_scale is not None:
            pa_mpjpe_value = pa_mpjpe_value * self.unit_scale

        masked_pa_mpjpe_value = pa_mpjpe_value[np.where(gt_mask > 0)]
        if self.outlier_threshold is not None:
            masked_pa_mpjpe_value = masked_pa_mpjpe_value[np.where(
                masked_pa_mpjpe_value < self.outlier_threshold)]

        pa_mpjpe_mean, pa_mpjpe_std = \
            np.mean(masked_pa_mpjpe_value), np.std(masked_pa_mpjpe_value)

        return dict(
            pa_mpjpe_mean=pa_mpjpe_mean,
            pa_mpjpe_std=pa_mpjpe_std,
            pa_mpjpe_value=pa_mpjpe_value)
