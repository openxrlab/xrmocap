# yapf: disable
import logging
import numpy as np
import os
import prettytable
import time
import torch
from torch.utils.data import DataLoader
from typing import List, Union

from xrmocap.data.data_visualization.builder import BaseDataVisualization
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.architecture.base_architecture import BaseArchitecture
from xrmocap.utils.distribute_utils import collect_results, is_main_process
from xrmocap.utils.eval_utils import align_convention_mask
from xrmocap.utils.mvp_utils import (
    AverageMeter, convert_result_to_kps, norm2absolute,
)
from .base_evaluation import BaseEvaluation
from .metrics.base_metric import BaseMetric

# yapf: enable


class End2EndEvaluation(BaseEvaluation):

    def __init__(
        self,
        test_loader: DataLoader,
        output_dir: str,
        metric_list: List[Union[dict, BaseMetric]],
        trans_matrix: Union[List[List[float]], None] = None,
        checkpoint_select: str = 'pcp_total_mean',
        n_max_person: int = 10,
        dataset_name: Union[None, str] = None,
        pred_kps3d_convention: str = 'coco',
        gt_kps3d_convention: str = 'coco',
        eval_kps3d_convention: str = 'human_data',
        pick_dict: Union[dict, None] = None,
        print_freq: int = 100,
        dataset_visualization: Union[None, dict, BaseDataVisualization] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Evaluation for end-to-end methods. For model validation during
        training and model evaluation.

        Args:
            test_loader (DataLoader):
                Test dataloader.
            output_dir (str):
                Output directory.
            metric_list (List[Union[dict, BaseMetric]]):
                A list of metrics to be evaluated.
            trans_matrix (Union[List[List[float]], None], optional):
                A rotation matrix to transform the ground truth world
                coordinate to align with prediction world coordinate.
            checkpoint_select (str, optional):
                Name of metric in the metric list to be used for
                checkpoint selection. Defaults to 'pcp_total_mean'.
            n_max_person (int, optional):
                Number of maximum person the model can predict.
                Defaults to 10.
            dataset_name (Union[None, str], optional):
                Name of the dataset. Defaults to None.
            pred_kps3d_convention (str, optional):
                Keypoint convention of predicted keypoints3d.
                Defaults to 'coco'.
            gt_kps3d_convention (str, optional):
                Keypoint convention of ground-truth keypoints3d.
                Defaults to 'coco'.
            eval_kps3d_convention (str, optional):
                A keypoint convention used for evaluation.
                It is recommended to use human_data if pred_kps3d_convention
                and gt_kps3d_convention are different.
                Defaults to 'human_data'.
            pick_dict (Union[dict, None], optional):
                Selected metrics to be printed in the final table.
                Defaults to None.
            print_freq (int, optional):
                Printing frequencty during model validation. Defaults to 100.
            dataset_visualization (Union[None, dict,
                BaseDataVisualization], optional):
                Dataset visualization. Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """

        BaseEvaluation.__init__(
            self,
            dataset=test_loader.dataset,
            output_dir=output_dir,
            metric_list=metric_list,
            pick_dict=pick_dict,
            dataset_visualization=dataset_visualization,
            eval_kps3d_convention=eval_kps3d_convention,
            logger=logger)

        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.print_freq = print_freq
        self.n_frame = test_loader.dataset.len
        self.pred_kps3d_convention = pred_kps3d_convention
        self.n_max_person = n_max_person
        self.gt_kps3d_convention = gt_kps3d_convention
        self.checkpoint_select = checkpoint_select
        self.trans_matrix = np.array(trans_matrix)

    def run(self,
            model: BaseArchitecture,
            threshold: float = 0.1,
            is_train: bool = False,
            overwrite: bool = False) -> float:
        """Run the evaluation.

        Args:
            model (BaseArchitecture):
                Model to be evaluated.
            threshold (float, optional):
                Threshold for the predicted confidence.
                Defaults to 0.1.
            is_train (bool, optional):
                If is model validation in training, no keypoints or
                inference results will be saved. Defaults to False.
            overwrite (bool, optional):
                Overwrite the output folder. Defaults to False.

        Returns:
            accuracy: selected metric evaluation result.
        """

        # prepare output folder
        BaseEvaluation.run(self, overwrite=overwrite)

        # get predicted Keypoints3d from model
        preds_single = self.model_validate(model, threshold=threshold)
        pred_kps3d_list = collect_results(preds_single, len(self.dataset))

        accuracy = None
        if is_main_process():
            # prepare predicted Keypoints3d
            # Add approximated points and update masks here if needed
            # e.g. nose, jaw, headtop
            n_frame = len(pred_kps3d_list)
            pred_n_kps = pred_kps3d_list[0].shape[1]
            pred_kps3d = np.full((n_frame, self.n_max_person, pred_n_kps, 4),
                                 np.nan)

            for frame_idx, per_frame_kps3d in enumerate(pred_kps3d_list):
                if len(per_frame_kps3d) > 0:
                    n_valid_person, keypoints3d_pred_valid = \
                        convert_result_to_kps([per_frame_kps3d])
                    pred_kps3d[
                        frame_idx, :n_valid_person] = keypoints3d_pred_valid

            pred_keypoints3d_raw = Keypoints(
                dtype='numpy',
                kps=pred_kps3d,
                mask=pred_kps3d[..., -1] > 0,
                convention=self.pred_kps3d_convention,
                logger=self.logger)

            # prepare gt Keypoints3d
            gt_n_person = np.max(
                np.array([
                    scene_keypoints.get_person_number()
                    for scene_keypoints in self.dataset.gt3d
                ]))
            gt_n_kps = self.dataset.gt3d[0].get_keypoints_number()
            gt_kps3d = np.full((self.n_frame, gt_n_person, gt_n_kps, 4),
                               np.nan)
            gt_mask = np.full((self.n_frame, gt_n_person, gt_n_kps), np.nan)

            start_frame = 0
            for gt_keypoints3d_scene in self.dataset.gt3d:

                gt_kps3d_scene = gt_keypoints3d_scene.get_keypoints().numpy(
                )  # [n_frame, n_person, n_kps, 4]
                gt_mask_scene = gt_keypoints3d_scene.get_mask().numpy(
                )  # [n_frame, n_person, n_kps]

                n_frame, n_person, _, _ = gt_kps3d_scene.shape
                end_frame = min(start_frame + n_frame, self.n_frame)

                gt_kps3d[start_frame:end_frame, :n_person, ...] = \
                    gt_kps3d_scene[:end_frame-start_frame, ...]
                gt_mask[start_frame:end_frame, :n_person, ...] = \
                    gt_mask_scene[:end_frame-start_frame, ...]
                start_frame = end_frame

            if self.dataset_name == 'panoptic' and \
                    self.trans_matrix is not None:
                for frame_idx in range(self.n_frame):
                    for person_idx in range(gt_n_person):
                        gt_kps3d[frame_idx, person_idx, :, :3] = \
                            gt_kps3d[frame_idx, person_idx, :, :3].dot(
                                self.trans_matrix)

            gt_keypoints3d_raw = Keypoints(
                dtype='numpy',
                kps=gt_kps3d,
                mask=gt_mask,
                convention=self.gt_kps3d_convention,
                logger=self.logger)

            # convert pred and gt to the same convention before passing
            # to metric manager, human_data recommended
            gt_keypoints3d, pred_keypoints3d = \
                align_convention_mask(pred_keypoints3d_raw,
                                      gt_keypoints3d_raw,
                                      self.pred_kps3d_convention,
                                      self.gt_kps3d_convention,
                                      self.eval_kps3d_convention,
                                      self.logger)

            # evaluate and print results
            eval_results, full_results = self.metric_manager(
                pred_keypoints3d=pred_keypoints3d,
                gt_keypoints3d=gt_keypoints3d)

            table = prettytable.PrettyTable()
            table.field_names = ['Metric name', 'Value']
            for metric_name, metric_dict in eval_results.items():
                for key, value in metric_dict.items():
                    table.add_row([f'{metric_name}: {key}', f'{value:.2f}'])
            table_str = '\n' + table.get_string()
            self.logger.info(table_str)

            if self.checkpoint_select in full_results:
                accuracy = full_results[self.checkpoint_select]
            else:
                self.logger.warning(
                    'Metric for checkpoint selection '
                    f'{self.checkpoint_select} is not available, '
                    'This may cause troubles in training,'
                    'please check the config.')
                if is_train:
                    raise KeyError

            # save and visualize when it is not train
            if not is_train:
                pred_file_path = os.path.join(self.output_dir,
                                              'pred_keypoints3d.npz')
                gt_file_path = os.path.join(self.output_dir,
                                            'gt_keypoints3d.npz')

                self.logger.info(f'Saving 3D keypoints to: {pred_file_path}')
                pred_keypoints3d.dump(pred_file_path)
                self.logger.info(f'Saving 3D keypoints to: {gt_file_path}')
                gt_keypoints3d.dump(gt_file_path)

                if self.dataset_visualization is not None:
                    self.dataset_visualization.pred_kps3d_paths = \
                        pred_file_path
                    self.dataset_visualization.run(overwrite=overwrite)

        return accuracy

    def model_validate(self, model: BaseArchitecture, threshold: float):
        """Validate model during training or testing.

        Args:
            model (BaseArchitecture):
                Model to be evaluated.
            threshold (float):
                Confidence threshold to filter non-human keypoints.
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
