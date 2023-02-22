# yapf: disable
import logging
import numpy as np
from prettytable import PrettyTable
from typing import List, Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from .base_metric import BaseMetric

# yapf: enable


class PrecisionRecallMetric(BaseMetric):
    """Precision and recall with given thrsholds. If the number of prediction
    does not align with the number of ground truth, this metric will evaluate
    based on the ground truth matched to the predictions.

    This is a rank-2 metric It depends on rank-1 metric MPJPE.
    """
    RANK = 2

    def __init__(
        self,
        name: str,
        threshold: Union[List[int], List[float]] = [25, 100],
        show_table: bool = False,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init precision and recall metric evaluation.

        Args:
            name (str):
                Name of the metric.
            threshold (Union[List[int],List[float]], optional):
                List of thresholds. Defaults to [25,100].
            show_table (bool, optional):
                Whether to show the table of detailed metric results.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """
        BaseMetric.__init__(self, name=name, logger=logger)
        self.threshold = threshold
        if 25 not in self.threshold:
            self.threshold.append(25)
            self.threshold.sort()
        self.show_table = show_table

    def __call__(self, pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                 **kwargs):
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
            self.n_frame = gt_n_frame
        else:
            self.logger.error('Prediction and ground-truth does not match in '
                              'the number of frame.')
            raise ValueError

        if 'match_matrix_gt2pred' in kwargs:
            self.match_matrix_pred2gt = kwargs['match_matrix_pred2gt']
        else:
            self.logger.error('No matching matrix found. '
                              'Please add PredictionMatcher in the config.')
            raise KeyError

        if 'mpjpe_value_pred2gt' in kwargs:
            self.mpjpe_value_pred2gt = kwargs['mpjpe_value_pred2gt']
        else:
            self.logger.error('No matching matrix found. '
                              'Please add PredictionMatcher in the config.')
            raise KeyError

        tb, precision_recall_dict = self.evaluate_map(pred_keypoints3d,
                                                      gt_keypoints3d)
        if self.show_table:
            self.logger.info('Detailed table for PrecisionRecallMetric\n' +
                             tb.get_string())

        return precision_recall_dict

    def evaluate_map(
        self,
        pred_keypoints3d: Keypoints,
        gt_keypoints3d: Keypoints,) \
            -> Tuple[List[float], List[float], float, float]:
        """Evaluate mAP and recall based on MPJPE.

        Args:
            pred_keypoints3d (Keypoints):
                Predicted 3D keypoints.
            threshold (float):
                Threshold for valid keypoints. Defaults to 0.1.

        Returns:
            Tuple[List[float], List[float], float, float]:
                List of AP, list of recall, MPJPE value and recall@500mm.
        """
        gt_n_person = gt_keypoints3d.get_keypoints().shape[1]
        pred_n_person = pred_keypoints3d.get_keypoints().shape[1]

        total_valid_gt = 0
        eval_list = []
        for frame_idx in range(self.n_frame):
            for person_idx in range(gt_n_person):
                gt_person_mask = gt_keypoints3d.get_mask()[frame_idx,
                                                           person_idx, ...]
                # skip invalid personin gt
                if (gt_person_mask == 0).all():
                    continue
                else:
                    total_valid_gt += 1

            for person_idx in range(pred_n_person):
                person_mask = pred_keypoints3d.get_mask()[frame_idx,
                                                          person_idx, :]
                person_score = \
                    pred_keypoints3d.get_keypoints()[
                        frame_idx, person_idx, :, -1][
                            np.where(person_mask > 0)].mean()
                person_mpjpe = self.mpjpe_value_pred2gt[frame_idx, person_idx][
                    np.where(person_mask > 0)].mean()
                gt_id_acc = frame_idx * gt_n_person + \
                    self.match_matrix_pred2gt[frame_idx, person_idx]

                eval_list.append({
                    'mpjpe': float(person_mpjpe),
                    'score': float(person_score),
                    'gt_id': int(gt_id_acc)
                })

        aps = []
        recs = []
        for t in self.threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_valid_gt, t)
            aps.append(ap)
            recs.append(rec)

        tb = PrettyTable()
        tb.field_names = ['Recall threshold/mm'] + \
            [f'{i}' for i in self.threshold]
        tb.add_row(['AP'] +
                   [f'{aps[i] * 100:.2f}' for i in range(len(self.threshold))])
        tb.add_row(
            ['Recall'] +
            [f'{recs[i] * 100:.2f}' for i in range(len(self.threshold))])

        precision_recall_dict = {}
        for i in range(len(self.threshold)):
            precision_recall_dict[
                f'recall@{self.threshold[i]}'] = recs[i] * 100
            precision_recall_dict[f'ap@{self.threshold[i]}'] = aps[i] * 100

        return tb, precision_recall_dict

    def _eval_list_to_ap(self, eval_list, total_valid_gt, threshold):
        """convert evaluation result to ap."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_valid_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]
