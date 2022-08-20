import numpy as np
import torch
from typing import List, Tuple

from xrmocap.data.dataset.base_dataset import BaseDataset


class MVPEvaluation:

    def __init__(self, dataset: BaseDataset, kps_thr: float = 0.1):
        self.dataset = dataset
        self.gt_num = dataset.len
        self.n_views = dataset.n_views
        self.kps_thr = kps_thr

    def evaluate_map(self, pred_kps3d: torch.Tensor) \
            -> Tuple[List[float], List[float], float, float]:
        """Evaluate MPJPE, mAP and recall based on MPJPE. Mainly for panoptic
        predictions.

        Args:
            pred_kps3d (torch.Tensor):
                Predicted 3D keypoints.

        Returns:
            Tuple[List[float], List[float], float, float]:
                List of AP, list of recall, MPJPE value and recall@500mm.
        """

        eval_list = []
        assert len(pred_kps3d) == self.gt_num, \
            f'number mismatch {len(pred_kps3d)} pred and {self.gt_num} gt'

        trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                     [0.0, 1.0, 0.0]]).double()

        total_gt = 0
        for index in range(self.gt_num):
            scene_idx, frame_idx, _ = self.dataset.process_index_mapping(index)
            gt_keypoints3d = self.dataset.gt3d[scene_idx]
            gt_scene_kps3d = gt_keypoints3d.get_keypoints(
            )  # [n_frame, n_person, n_kps, 4]
            gt_frame_kps3d = gt_scene_kps3d[
                frame_idx][:, :self.dataset.n_kps, :]  # [n_person, n_kps, 4]

            check_valid = torch.sum(gt_frame_kps3d, axis=1)  # [n_person, 4]
            gt_frame_kps3d = gt_frame_kps3d[check_valid[:, -1] > 0]

            gt_frame_kps3d = [
                torch.mm(gt_person_kps3d[:, 0:3], trans_ground)
                for gt_person_kps3d in gt_frame_kps3d
            ]
            if len(gt_frame_kps3d) == 0:
                continue

            pred_frame_kps3d = pred_kps3d[index].copy()
            pred_frame_kps3d_valid = pred_frame_kps3d[pred_frame_kps3d[:, 0,
                                                                       3] >= 0]

            for pred_person_kps3d in pred_frame_kps3d_valid:
                mpjpes = []
                for gt_person_kps3d in gt_frame_kps3d:
                    vis = gt_person_kps3d[:, -1] > self.kps_thr

                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum(
                                (np.array(pred_person_kps3d[vis, 0:3]) -
                                 np.array(gt_person_kps3d[vis, 0:3]))**2,
                                axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pred_person_kps3d[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += len(gt_frame_kps3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return \
            aps, \
            recs, \
            self._eval_list_to_mpjpe(eval_list), \
            self._eval_list_to_recall(eval_list, total_gt)

    def evaluate_pcp(self,
                     pred_kps3d: torch.Tensor,
                     recall_threshold: int = 500,
                     alpha: float = 0.5) -> Tuple[List[float], float, float]:
        """Evaluate MPJPE and PCP. Mainly for Shelf and Campus predictions.

        Args:
            pred_kps3d (torch.Tensor):
                Predicted 3D keypoints.
            recall_threshold (int, optional):
                Threshold for MPJPE. Defaults to 500.
            alpha (float, optional):
                Threshold for correct limb part. Defaults to 0.5.
                Predicted limb part is regarded as correct if
                predicted_part_length < alpha * gt_part_length
        Returns:
            Tuple[List[float], float, float]:
                List of PCP per actor, average PCP, and recall@500mm.
        """
        assert len(pred_kps3d) == self.gt_num, \
            f'number mismatch {len(pred_kps3d)} pred and {self.gt_num} gt'

        trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]]).double()

        # TODO: can get this info from Keypoints?
        limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10],
                 [10, 11], [12, 13]]

        gt_n_person = np.max(
            np.array([
                scene_keypoints.get_keypoints().shape[1]
                for scene_keypoints in self.dataset.gt3d
            ]))

        correct_parts = np.zeros(gt_n_person)
        total_parts = np.zeros(gt_n_person)
        limb_correct_parts = np.zeros((gt_n_person, 10))

        total_gt = 0
        match_gt = 0
        for index in range(self.gt_num):
            scene_idx, frame_idx, _ = self.dataset.process_index_mapping(index)
            gt_keypoints3d = self.dataset.gt3d[scene_idx]
            gt_scene_kps3d = gt_keypoints3d.get_keypoints(
            )  # [n_frame, n_person, n_kps, 4]
            gt_frame_kps3d = gt_scene_kps3d[
                frame_idx][:, :self.dataset.n_kps, :]  # [n_person, n_kps, 4]

            gt_frame_kps3d = [
                torch.mm(gt_person_kps3d[:, 0:3], trans_ground)
                for gt_person_kps3d in gt_frame_kps3d
            ]
            if len(gt_frame_kps3d) == 0:
                continue

            pred_frame_kps3d = pred_kps3d[index].copy()
            pred_frame_kps3d_valid = pred_frame_kps3d[pred_frame_kps3d[:, 0, 3]
                                                      >= 0]  # if is a person

            for person_idx, gt_person_kps3d in enumerate(gt_frame_kps3d):

                vis = gt_person_kps3d[:, -1] > self.kps_thr

                check_valid = torch.sum(gt_person_kps3d, axis=1)  # [4]
                if check_valid[-1] == 0:
                    continue

                mpjpes = np.mean(
                    np.sqrt(
                        np.sum(
                            (np.array(pred_frame_kps3d_valid[:, vis, 0:3]) -
                             np.array(gt_person_kps3d[vis, 0:3]))**2,
                            axis=-1)),
                    axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)

                if min_mpjpe < recall_threshold:
                    match_gt += 1

                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person_idx] += 1
                    error_s = \
                        np.linalg.norm(np.array(
                            pred_frame_kps3d_valid[min_n, k[0], 0:3]) -
                            np.array(gt_person_kps3d[k[0]]))
                    error_e = \
                        np.linalg.norm(np.array(
                            pred_frame_kps3d_valid[min_n, k[1], 0:3]) -
                            np.array(gt_person_kps3d[k[1]]))
                    limb_length = np.linalg.norm(
                        np.array(gt_person_kps3d[k[0]]) -
                        np.array(gt_person_kps3d[k[1]]))

                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person_idx] += 1
                        limb_correct_parts[person_idx, j] += 1

                # TODO: get these kps inedex from kps name
                pred_hip = (pred_frame_kps3d_valid[min_n, 2, 0:3] +
                            pred_frame_kps3d_valid[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt_person_kps3d[2] + gt_person_kps3d[3]) / 2.0
                total_parts[person_idx] += 1
                error_s = np.linalg.norm(np.array(pred_hip) - np.array(gt_hip))
                error_e = np.linalg.norm(
                    np.array(pred_frame_kps3d_valid[min_n, 12, 0:3]) -
                    np.array(gt_person_kps3d[12]))
                limb_length = np.linalg.norm(
                    np.array(gt_hip) - np.array(gt_person_kps3d[12]))

                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person_idx] += 1
                    limb_correct_parts[person_idx, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        return \
            actor_pcp, avg_pcp, match_gt / (total_gt + 1e-8)

    def _eval_list_to_ap(self, eval_list, total_gt, threshold):
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
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    def _eval_list_to_mpjpe(self, eval_list, threshold=500):
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    def _eval_list_to_recall(self, eval_list, total_gt, threshold=500):
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt
