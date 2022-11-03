# yapf: disable
import logging
import numpy as np
import os
import time
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from typing import List, Tuple, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.core.evaluation.align_keypoints3d import align_keypoints3d
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.architecture.base_architecture import BaseArchitecture
from xrmocap.utils.distribute_utils import collect_results, is_main_process
from xrmocap.utils.mvp_utils import (
    AverageMeter, convert_result_to_kps, norm2absolute,
)
from .metrics import calc_limbs_accuracy, evaluate

# yapf: enable


class MVPEvaluation:
    """Evaluation for MvP method."""

    def __init__(
        self,
        test_loader: DataLoader,
        selected_limbs_name: List[List[str]] = [],
        additional_limbs_names: List[List[str]] = [],
        n_max_person: int = 10,
        dataset_name: Union[None, str] = None,
        pred_kps3d_convention: str = 'coco',
        gt_kps3d_convention: str = 'coco',
        print_freq: int = 100,
        final_output_dir: Union[None, str] = None,
        logger: Union[None, str, logging.Logger] = None,
    ):
        """Initialization for the class.

        Args:
            test_loader (DataLoader):
                Test dataloader.
            selected_limbs_name (List[List[str]], optional):
                The name of selected
                limbs in evaluation. Defaults to [].
            additional_limbs_names (List[List[str]], optional):
                Names at both ends of the limb. Defaults to [].
            n_max_person (int, optional):
                Number of maximum person the model can predict. Defaults to 10.
            dataset_name (Union[None, str], optional):
                Name of the dataset. Defaults to None.
            pred_kps3d_convention (str, optional):
                Keypoints3d convention of the predicted keypoints.
                Defaults to 'coco'.
            gt_kps3d_convention (str, optional):
                Keypoints3d convention of the ground-truth keypoints.
                Defaults to 'coco'.
            print_freq (int, optional):
                Printing frequency. Defaults to 100.
            final_output_dir (Union[None, str], optional):
                Directory to output folder. Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """

        self.logger = get_logger(logger)

        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.print_freq = print_freq
        self.output_dir = final_output_dir
        self.dataset = test_loader.dataset
        self.gt_n_frame = test_loader.dataset.len
        self.pred_kps3d_convention = pred_kps3d_convention
        self.n_max_person = n_max_person
        self.selected_limbs_name = selected_limbs_name
        self.additional_limbs_names = additional_limbs_names
        self.gt_kps3d_convention = gt_kps3d_convention

    def run(
        self,
        model: BaseArchitecture,
        threshold: float = 0.1,
        is_train: bool = False,
    ):

        # validate model and get predictions
        preds_single = self.model_validate(
            model,
            threshold=threshold,
            is_train=is_train,
        )
        kps3d_pred_list = collect_results(preds_single, len(self.dataset))

        # save predicted keypoints3d
        precision = None
        if is_main_process():
            n_frame = len(kps3d_pred_list)
            n_kps = kps3d_pred_list[0].shape[1]
            kps3d_pred = np.full((n_frame, self.n_max_person, n_kps, 4),
                                 np.nan)

            for frame_idx, per_frame_kps3d in enumerate(kps3d_pred_list):
                if len(per_frame_kps3d) > 0:
                    n_valid_person, keypoints3d_pred_valid = \
                        convert_result_to_kps([per_frame_kps3d])
                    kps3d_pred[
                        frame_idx, :n_valid_person] = keypoints3d_pred_valid

            keypoints3d_pred = Keypoints(
                dtype='numpy',
                kps=kps3d_pred,
                mask=kps3d_pred[..., -1] > 0,
                convention=self.pred_kps3d_convention,
                logger=self.logger)

            # convert gt to keypoints3d
            gt_n_person = np.max(
                np.array([
                    scene_keypoints.get_person_number()
                    for scene_keypoints in self.dataset.gt3d
                ]))
            gt_n_kps = self.dataset.gt3d[0].get_keypoints_number()
            kps3d_gt = np.full((self.gt_n_frame, gt_n_person, gt_n_kps, 4),
                               np.nan)
            mask_gt = np.full((self.gt_n_frame, gt_n_person, gt_n_kps), np.nan)

            start_frame = 0
            for keypoints3d_gt_scene in self.dataset.gt3d:

                kps3d_gt_scene = keypoints3d_gt_scene.get_keypoints().numpy(
                )  # [n_frame, n_person, n_kps, 4]
                mask_gt_scene = keypoints3d_gt_scene.get_mask().numpy(
                )  # [n_frame, n_person, n_kps]

                n_frame, n_person, _, _ = kps3d_gt_scene.shape
                end_frame = min(start_frame + n_frame, self.gt_n_frame)

                kps3d_gt[start_frame:end_frame, :n_person, ...] = \
                    kps3d_gt_scene[:end_frame-start_frame, ...]
                mask_gt[start_frame:end_frame, :n_person, ...] = \
                    mask_gt_scene[:end_frame-start_frame, ...]
                start_frame = end_frame

            keypoints3d_gt = Keypoints(
                dtype='numpy',
                kps=kps3d_gt,
                mask=mask_gt,
                convention=self.gt_kps3d_convention,
                logger=self.logger)

            if not is_train:
                pred_file = os.path.join(self.output_dir, 'pred_kps3d.npz')
                gt_file = os.path.join(self.output_dir, 'gt_kps3d.npz')

                self.logger.info(f'Saving 3D keypoints to: {pred_file}')
                keypoints3d_pred.dump(pred_file)
                self.logger.info(f'Saving 3D keypoints to: {gt_file}')
                keypoints3d_gt.dump(gt_file)

            # quantitative evaluation and print result
            if 'panoptic' in self.dataset_name:
                tb = PrettyTable()
                mpjpe_threshold = np.arange(25, 155, 25)

                aps, recs, mpjpe, recall500 = \
                    self.evaluate_map(keypoints3d_pred, keypoints3d_gt)

                tb.field_names = ['Threshold/mm'] + \
                    [f'{i}' for i in mpjpe_threshold]
                tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
                tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
                tb.add_row(['recall@500mm'] +
                           [f'{recall500 * 100:.2f}' for re in recs])
                self.logger.info('\n' + tb.get_string())
                self.logger.info(f'MPJPE: {mpjpe:.2f}mm')

                precision = np.mean(aps[0])

            elif 'campus' in self.dataset_name \
                    or 'shelf' in self.dataset_name:

                pred_keypoints3d_, gt_keypoints3d_, limbs = align_keypoints3d(
                    keypoints3d_pred, keypoints3d_gt,
                    self.pred_kps3d_convention, self.selected_limbs_name,
                    self.additional_limbs_names)
                check_result, eval_table = calc_limbs_accuracy(
                    pred_keypoints3d_,
                    gt_keypoints3d_,
                    limbs,
                    logger=self.logger)

                person_wise_avg = np.sum(
                    check_result > 0, axis=(0, 2)) / np.sum(
                        np.abs(check_result), axis=(0, 2))
                precision = np.sum(person_wise_avg) / len(person_wise_avg)

                self.logger.info('\n' + eval_table.get_string())
                evel_dict = evaluate(
                    pred_keypoints3d_,
                    gt_keypoints3d_,
                    scale=1.,
                    logger=self.logger)
                self.logger.info('MPJPE: {:.2f} ± {:.2f} mm'.format(
                    evel_dict['mpjpe_mean'], evel_dict['mpjpe_std']))
                self.logger.info(
                    f'PA-MPJPE: {evel_dict["pa_mpjpe_mean"]:.2f} ± '
                    f'{evel_dict["pa_mpjpe_std"]:.2f} mm')
                self.logger.info(f'PCK@50mm: {evel_dict["pck"][50]:.2f} %')
                self.logger.info(f'PCK@100mm: {evel_dict["pck"][100]:.2f} %')

            else:
                self.logger.warning(f'Dataset {self.dataset_name} '
                                    'is not yet implemented.')
                raise NotImplementedError

        return precision

    def model_validate(self,
                       model: BaseArchitecture,
                       threshold: float,
                       is_train: bool = False):
        """Validate model during training or testing.

        Args:
            model (BaseArchitecture):
                Model to be evaluated.
            threshold (float):
                Confidence threshold to filter non-human keypoints.
            is_train (bool, optional):
                True if it is called during trainig. Defaults to False.

        Returns:
            float: model evaluation result, precision
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.eval()

        preds = []
        with torch.no_grad():
            end = time.time()
            for i, (inputs, meta) in enumerate(self.test_loader):
                data_time.update(time.time() - end)
                assert len(inputs) == self.dataset.n_views
                output = model(views=inputs, meta=meta)

                gt_kps3d = meta[0]['kps3d'].float()
                n_kps = gt_kps3d.shape[2]
                bs, n_queries = output['pred_logits'].shape[:2]

                src_poses = output['pred_poses']['outputs_coord']. \
                    view(bs, n_queries, n_kps, 3)
                src_poses = norm2absolute(src_poses, model.module.grid_size,
                                          model.module.grid_center)
                score = output['pred_logits'][:, :, 1:2].sigmoid()
                score = score.unsqueeze(2).expand(-1, -1, n_kps, -1)
                temp = (score > threshold).float() - 1

                pred = torch.cat([src_poses, temp, score], dim=-1)
                pred = pred.detach().cpu().numpy()
                for b in range(pred.shape[0]):
                    preds.append(pred[b])

                batch_time.update(time.time() - end)
                end = time.time()
                if (i % self.print_freq == 0 or i
                        == len(self.test_loader) - 1) and is_main_process():
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    speed = len(inputs) * inputs[0].size(0) / batch_time.val
                    msg = f'Test: [{i}/{len(self.test_loader)}]\t' \
                        f'Time: {batch_time.val:.3f}s ' \
                        f'({batch_time.avg:.3f}s)\t' \
                        f'Speed: {speed:.1f} samples/s\t' \
                        f'Data: {data_time.val:.3f}s ' \
                        f'({data_time.avg:.3f}s)\t' \
                        f'Memory {gpu_memory_usage:.1f}'
                    self.logger.info(msg)

        return preds

    def evaluate_map(
        self,
        pred_keypoints3d: Keypoints,
        gt_keypoints3d: Keypoints,
        threshold: float = 0.1) \
            -> Tuple[List[float], List[float], float, float]:
        """Evaluate MPJPE, mAP and recall based on MPJPE. Mainly for panoptic
        predictions.

        Args:
            pred_keypoints3d (Keypoints):
                Predicted 3D keypoints.
            threshold (float):
                Threshold for valid keypoints. Defaults to 0.1.

        Returns:
            Tuple[List[float], List[float], float, float]:
                List of AP, list of recall, MPJPE value and recall@500mm.
        """

        pred_kps3d = pred_keypoints3d.get_keypoints()
        gt_kps3d = gt_keypoints3d.get_keypoints()
        gt_n_frame, _, _, _ = gt_kps3d.shape
        pred_n_frame, _, n_kps, _ = pred_kps3d.shape
        total_gt = 0
        eval_list = []
        if not pred_n_frame == gt_n_frame:
            self.logger.error(f'Frame number mismatch: {pred_n_frame} '
                              f'pred and {self.gt_n_frame} gt')
            raise ValueError

        trans_ground = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                 [0.0, 1.0, 0.0]])

        for frame_idx_all_scenes in range(gt_n_frame):

            gt_frame_kps3d = gt_kps3d[frame_idx_all_scenes, :, :n_kps, :]
            check_valid = np.sum(gt_frame_kps3d, axis=1)  # [n_person, 4]
            gt_frame_kps3d = gt_frame_kps3d[check_valid[:, -1] > 0]

            gt_frame_kps3d = [
                gt_person_kps3d[:, 0:3].dot(trans_ground)
                for gt_person_kps3d in gt_frame_kps3d
            ]
            if len(gt_frame_kps3d) == 0:
                continue

            pred_frame_kps3d_valid = pred_kps3d[frame_idx_all_scenes].copy()

            for pred_person_kps3d in pred_frame_kps3d_valid:
                mpjpes = []
                for gt_person_kps3d in gt_frame_kps3d:
                    vis = gt_person_kps3d[:, -1] > threshold

                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum(
                                (np.array(pred_person_kps3d[vis, 0:3]) -
                                 np.array(gt_person_kps3d[vis, 0:3]))**2,
                                axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pred_person_kps3d[0, -1]
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

    def _eval_list_to_ap(self, eval_list, total_gt, threshold):
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
        """convert evaluation result to mpjpe."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    def _eval_list_to_recall(self, eval_list, total_gt, threshold=500):
        """convert evaluation result to recall."""
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt
