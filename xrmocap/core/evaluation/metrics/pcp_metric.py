# yapf: disable
import logging
import numpy as np
from prettytable import PrettyTable
from typing import List, Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx
from xrmocap.transform.limbs import get_limbs_from_keypoints
from xrmocap.utils.mvpose_utils import check_limb_is_correct
from .base_metric import BaseMetric

# yapf: enable


class PCPMetric(BaseMetric):
    """Percentage of Correct Parts (PCP) metric measures percentage of the
    corrected predicted limbs.

    This is a rank-1 metric, depends on rank-0 metric PredictionMatcher. If
    thenumber of prediction does not align with the number of ground truth,
    this metric will evaluate predictions matched to the ground truth.
    """
    RANK = 1

    def __init__(
        self,
        name: str,
        selected_limbs_names: List[str],
        threshold: float = 0.5,
        additional_limbs_names: Union[None, List[str]] = None,
        show_table: bool = False,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Init PCP metric evaluation.

        Args:
            name (str):
                Name of the metric.
            threshold (float):
                Threshold for correct limb. If the error is less than
                threshold * limb_length, limb is considered as correct.
                Defaults to 0.5.
            selected_limbs_names (List[str]):
                List of selected limbs' name for PCP metric.
            additional_limbs_names (Union[None, List[str]], optional):
                Additional limbs to be evaluated .
                Defaults to None.
            show_table (bool, optional):
                Whether to show the table of detailed metric result .
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """

        BaseMetric.__init__(self, name=name, logger=logger)
        self.threshold = threshold
        self.selected_limbs_names = selected_limbs_names
        self.additional_limbs_names = additional_limbs_names
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

        gt_n_frame = gt_keypoints3d.get_keypoints().shape[0]
        pred_n_frame = pred_keypoints3d.get_keypoints().shape[0]
        if gt_n_frame == pred_n_frame:
            self.n_frame = gt_n_frame
        else:
            self.logger.error('Prediction and ground-truth does not match in '
                              'the number of frame.')
            raise ValueError

        if 'match_matrix_gt2pred' in kwargs:
            self.match_matrix_gt2pred = kwargs['match_matrix_gt2pred']
        else:
            self.logger.error('No matching metric found. '
                              'Please add PredictionMatcher in the config.')
            raise KeyError

        limbs, limb_name = self.get_limbs(pred_keypoints3d,
                                          self.selected_limbs_names,
                                          self.additional_limbs_names)

        pcp_mean, eval_table = \
            self.calc_limbs_accuracy(pred_keypoints3d, gt_keypoints3d,
                                     limbs, limb_name)
        if self.show_table:
            self.logger.info('Detailed table for PCPMetric\n' +
                             eval_table.get_string())

        return dict(pcp_total_mean=pcp_mean)

    def get_limbs(self, pred_keypoints3d: Keypoints,
                  selected_limbs_name: List[List[str]],
                  additional_limbs_names: List[List[str]]):
        """align keypoints convention.

        Args:
            pred_keypoints3d (Keypoints): prediction of keypoints
            selected_limbs_name (List): selected limbs to be evaluated
            additional_limbs_names (List): additional limbs to be evaluated
        """
        ret_limbs = []
        ret_limb_name = []
        limb_name_list = []
        conn_list = []

        limbs = get_limbs_from_keypoints(
            keypoints=pred_keypoints3d, fill_limb_names=True)
        for limb_name, conn in limbs.get_connections_by_names().items():
            limb_name_list.append(limb_name)
            conn_list.append(conn)

        for frame_idx, limb_name in enumerate(limb_name_list):
            if limb_name in selected_limbs_name:
                ret_limbs.append(conn_list[frame_idx])
                ret_limb_name.append(limb_name)
            else:
                self.logger.info(f'{limb_name.title()} is not selected!')

        if additional_limbs_names is not None:
            for conn_names in additional_limbs_names:
                kps_idx_0 = get_keypoint_idx(
                    name=conn_names[0], convention=self.convention)
                kps_idx_1 = get_keypoint_idx(
                    name=conn_names[1], convention=self.convention)
                ret_limbs.append(
                    np.array([kps_idx_0, kps_idx_1], dtype=np.int32))
                ret_limb_name.append(f'{conn_names[0]}-{conn_names[1]}')

        return ret_limbs, ret_limb_name

    def calc_limbs_accuracy(
        self,
        pred_keypoints3d: Keypoints,
        gt_keypoints3d: Keypoints,
        limbs: List[List[int]],
        limb_name: List[str],
    ) -> Tuple[float, PrettyTable]:
        """Calculate accuracy of given list of limbs.

        Args:
            pred_keypoints3d (Keypoints):
                Predicted keypoints3d.
            gt_keypoints3d (Keypoints):
                Ground-truth keypoints3d.
            limbs (List[List[int]]):
                List of limbs connection.
            limb_name (List[str]):
                List of limbs name.

        Returns:
            Tuple[float, PrettyTable]:
                Accuracy and table of detailed results.
        """

        n_gt_person = gt_keypoints3d.get_person_number()
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        gt_kps3d_mask = gt_keypoints3d.get_mask()
        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        check_result = np.zeros((self.n_frame, n_gt_person, len(limbs)),
                                dtype=np.int32)
        accuracy_cnt = 0
        error_cnt = 0

        for frame_idx in range(self.n_frame):
            if not gt_kps3d_mask[frame_idx].any():
                continue
            gt_kps3d_idxs = np.where(
                np.sum(gt_kps3d_mask[frame_idx], axis=1) > 0)[0]
            for gt_kps3d_idx in gt_kps3d_idxs:
                f_gt_kps3d = gt_kps3d[frame_idx][gt_kps3d_idx]
                f_pred_kps3d = pred_kps3d[frame_idx]
                f_pred_kps3d_idx = self.match_matrix_gt2pred[frame_idx][
                    gt_kps3d_idx]
                if len(f_pred_kps3d) == 0:
                    continue

                f_pred_kps3d = f_pred_kps3d[f_pred_kps3d_idx]

                for i, limb in enumerate(limbs):
                    start_point, end_point = limb
                    if check_limb_is_correct(f_pred_kps3d[start_point],
                                             f_pred_kps3d[end_point],
                                             f_gt_kps3d[start_point],
                                             f_gt_kps3d[end_point],
                                             self.threshold):
                        check_result[frame_idx, gt_kps3d_idx, i] = 1
                        accuracy_cnt += 1
                    else:
                        check_result[frame_idx, gt_kps3d_idx, i] = -1
                        error_cnt += 1

        bone_group = dict()
        for i, name in enumerate(limb_name):
            bone_group[name] = np.array([i])

        person_wise_avg = np.sum(
            check_result > 0, axis=(0, 2)) / np.sum(
                np.abs(check_result), axis=(0, 2))
        total_mean = np.sum(person_wise_avg) / len(person_wise_avg)

        bone_wise_result = dict()
        bone_person_wise_result = dict()
        for k, v in bone_group.items():
            bone_wise_result[k] = np.sum(check_result[:, :, v] > 0) / np.sum(
                np.abs(check_result[:, :, v]))
            bone_person_wise_result[k] = np.sum(
                check_result[:, :, v] > 0, axis=(0, 2)) / np.sum(
                    np.abs(check_result[:, :, v]), axis=(0, 2))

        tb = PrettyTable()
        tb.field_names = ['Bone Group'] + \
            [f'Actor {i}' for i in range(n_gt_person)] + ['Average']
        for k, v in bone_person_wise_result.items():
            this_row = [k] + [np.char.mod('%.2f', i * 100) for i in v] + [
                np.char.mod('%.2f', (np.sum(v) / len(v)) * 100)
            ]
            tb.add_row(this_row)
        this_row = ['total'] + [
            np.char.mod('%.2f', i * 100) for i in person_wise_avg
        ] + [np.char.mod('%.2f', total_mean * 100)]
        tb.add_row(this_row)

        return total_mean * 100, tb
