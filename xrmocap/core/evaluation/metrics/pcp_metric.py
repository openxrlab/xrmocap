# yapf: disable
import logging
import numpy as np
from prettytable import PrettyTable
from typing import List, Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx
from xrmocap.transform.limbs import get_limbs_from_keypoints
from xrmocap.utils.mvpose_utils import (
    check_limb_is_correct, vectorize_distance,
)
from .base_metric import BaseMetric

# yapf: enable


class PCPMetric(BaseMetric):
    """Percentage of Correct Parts (PCP) metric measures percentage of the
    corrected predicted limbs.

    This is a rank-0 metric.
    """
    RANK = 0

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

        limbs = self.get_limbs(pred_keypoints3d, gt_keypoints3d,
                               self.selected_limbs_names,
                               self.additional_limbs_names)

        pcp_mean, eval_table = \
            self.calc_limbs_accuracy(pred_keypoints3d, gt_keypoints3d,
                                     limbs)
        if self.show_table:
            self.logger.info('Detailed table for PCPMetric\n' +
                             eval_table.get_string())

        return dict(pcp_total_mean=pcp_mean)

    def get_limbs(self, pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                  selected_limbs_name: List[List[str]],
                  additional_limbs_names: List[List[str]]):
        """align keypoints convention.

        Args:
            pred_keypoints3d (Keypoints): prediction of keypoints
            gt_keypoints3d (Keypoints): ground true of keypoints
            eval_kps3d_convention (string): keypoints convention to align
            selected_limbs_name (List): selected limbs to be evaluated
            additional_limbs_names (List): additional limbs to be evaluated
        """
        ret_limbs = []
        limb_name_list = []
        conn_list = []

        limbs = get_limbs_from_keypoints(
            keypoints=pred_keypoints3d, fill_limb_names=True)
        for limb_name, conn in limbs.get_connections_by_names().items():
            limb_name_list.append(limb_name)
            conn_list.append(conn)

        for idx, limb_name in enumerate(limb_name_list):
            if limb_name in selected_limbs_name:
                ret_limbs.append(conn_list[idx])

        for conn_names in additional_limbs_names:
            kps_idx_0 = get_keypoint_idx(
                name=conn_names[0], convention=self.convention)
            kps_idx_1 = get_keypoint_idx(
                name=conn_names[1], convention=self.convention)
            ret_limbs.append(np.array([kps_idx_0, kps_idx_1], dtype=np.int32))

        return ret_limbs

    def calc_limbs_accuracy(
        self,
        pred_keypoints3d: Keypoints,
        gt_keypoints3d: Keypoints,
        limbs: List[List[int]],
    ) -> Tuple[float, PrettyTable]:
        """Calculate accuracy of given list of limbs.

        Args:
            pred_keypoints3d (Keypoints):
                Predicted keypoints3d.
            gt_keypoints3d (Keypoints):
                Ground-truth keypoints3d.
            limbs (List[List[int]]):
                List of limbs connection.

        Returns:
            Tuple[float, PrettyTable]:
                Accuracy and table of detailed results.
        """

        n_frame = gt_keypoints3d.get_frame_number()
        n_gt_person = gt_keypoints3d.get_person_number()
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        gt_kps3d_mask = gt_keypoints3d.get_mask()
        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        pred_kps3d_mask = pred_keypoints3d.get_mask()
        check_result = np.zeros((n_frame, n_gt_person, len(limbs) + 1),
                                dtype=np.int32)
        accuracy_cnt = 0
        error_cnt = 0

        for idx in range(n_frame):
            if not gt_kps3d_mask[idx].any():
                continue
            gt_kps3d_idxs = np.where(np.sum(gt_kps3d_mask[idx], axis=1) > 0)[0]
            for gt_kps3d_idx in gt_kps3d_idxs:
                f_gt_kps3d = gt_kps3d[idx][gt_kps3d_idx]
                f_pred_kps3d = pred_kps3d[idx][
                    np.sum(pred_kps3d_mask[idx], axis=1) > 0]
                if len(f_pred_kps3d) == 0:
                    continue

                dist = vectorize_distance(f_gt_kps3d[np.newaxis], f_pred_kps3d)
                f_pred_kps3d = f_pred_kps3d[np.argmin(dist[0])]

                for i, limb in enumerate(limbs):
                    start_point, end_point = limb
                    if check_limb_is_correct(f_pred_kps3d[start_point],
                                             f_pred_kps3d[end_point],
                                             f_gt_kps3d[start_point],
                                             f_gt_kps3d[end_point],
                                             self.threshold):
                        check_result[idx, gt_kps3d_idx, i] = 1
                        accuracy_cnt += 1
                    else:
                        check_result[idx, gt_kps3d_idx, i] = -1
                        error_cnt += 1
                gt_hip = (f_gt_kps3d[2] + f_gt_kps3d[3]) / 2
                pred_hip = (f_pred_kps3d[2] + f_pred_kps3d[3]) / 2
                if check_limb_is_correct(pred_hip, f_pred_kps3d[12], gt_hip,
                                         f_gt_kps3d[12], self.threshold):
                    check_result[idx, gt_kps3d_idx, -1] = 1
                    accuracy_cnt += 1
                else:
                    check_result[idx, gt_kps3d_idx, -1] = -1
                    error_cnt += 1
        bone_group = dict([('Torso', np.array([len(limbs) - 1])),
                           ('Upper arms', np.array([5, 6])),
                           ('Lower arms', np.array([4, 7])),
                           ('Upper legs', np.array([1, 2])),
                           ('Lower legs', np.array([0, 3]))])
        if len(limbs) > 8:
            # head is absent in some dataset
            bone_group['Head'] = np.array([8])

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
            [f'Actor {i}' for i in
                range(bone_person_wise_result['Torso'].shape[0])] + \
            ['Average']
        for k, v in bone_person_wise_result.items():
            this_row = [k] + [np.char.mod('%.2f', i * 100) for i in v] + [
                np.char.mod('%.2f', (np.sum(v) / len(v)) * 100)
            ]
            tb.add_row(this_row)
        this_row = ['Total'] + [
            np.char.mod('%.2f', i * 100) for i in person_wise_avg
        ] + [np.char.mod('%.2f', total_mean * 100)]
        tb.add_row(this_row)

        return total_mean * 100, tb
