# yapf: disable
import logging
import numpy as np
import os
import prettytable
from tqdm import tqdm
from typing import List, Union

from xrmocap.data.data_visualization.builder import BaseDataVisualization
from xrmocap.data.dataset.builder import MviewMpersonDataset
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_num,
)
from .base_evaluation import BaseEvaluation
from .metrics.base_metric import BaseMetric

# yapf: enable


class End2EndEvaluation(BaseEvaluation):

    def __init__(
        self,
        dataset: Union[dict, MviewMpersonDataset],
        output_dir: str,
        metric_list: List[Union[dict, BaseMetric]],
        pick_dict: Union[dict, None] = None,
        dataset_visualization: Union[None, dict, BaseDataVisualization] = None,
        eval_kps3d_convention: str = 'human_data',
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        BaseEvaluation.__init__(
            self,
            dataset=dataset,
            output_dir=output_dir,
            metric_list=metric_list,
            pick_dict=pick_dict,
            dataset_visualization=dataset_visualization,
            eval_kps3d_convention=eval_kps3d_convention,
            logger=logger)

    def run(self, overwrite: bool = False):
        BaseEvaluation.run(self, overwrite=overwrite)
        n_frame = len(self.dataset)
        # pred convention of the end2end model
        pred_convention = self.model.pred_kps3d_convention
        n_kps = get_keypoint_num(convention=pred_convention)
        pred_kps3d = np.zeros(shape=(n_frame, 1, n_kps, 4))
        gt_kps3d = None
        # where to stop recording the last scene
        end_of_clip_idxs = []
        # max number of person
        max_identity = 0
        identities = []
        for frame_idx, frame_item in enumerate(tqdm(self.dataset)):
            mview_img_tensor, _, _, _, kps3d, end_of_clip, kw_data = frame_item
            scene_idx = len(end_of_clip_idxs)
            # get fisheye param by scene_idx
            fisheye_list = self.dataset.fisheye_params[scene_idx]
            _ = len(fisheye_list)
            if end_of_clip:
                end_of_clip_idxs.append(frame_idx)
            # predict kps3d end2end
            pred_kps3d = None
            # concat predict kps3d
            for idx, identity in enumerate(identities):
                if identity > max_identity:
                    n_identity = identity - max_identity
                    pred_kps3d = np.concatenate(
                        (pred_kps3d,
                         np.zeros(shape=(n_frame, n_identity, n_kps, 4))),
                        axis=1)
                    max_identity = identity
                pred_kps3d[frame_idx, identity] = pred_kps3d
            # concat ground truth kps3d
            if gt_kps3d is None:
                gt_kps3d = kps3d.numpy()[np.newaxis]
            else:
                gt_kps3d = np.concatenate(
                    (gt_kps3d, kps3d.numpy()[np.newaxis]), axis=0)
        # if scenes stop at frame 3, 6,
        # split_idxs shall be [0, 4, 7, len(self.dataset)]
        split_idxs = (np.asarray(end_of_clip_idxs) + 1).tolist()
        split_idxs.insert(0, 0)
        if split_idxs[-1] != len(self.dataset):
            split_idxs.append(len(self.dataset))
        # save pred_kps3d_paths for visualization
        pred_kps3d_paths = []
        for scene_idx in range(len(split_idxs) - 1):
            start_idx = split_idxs[scene_idx]
            end_idx = split_idxs[scene_idx + 1]
            scene_pred_kps3d = pred_kps3d[start_idx:end_idx]
            pred_keypoints3d = Keypoints(
                kps=scene_pred_kps3d,
                mask=scene_pred_kps3d[..., -1:],
                convention=pred_convention,
                logger=self.logger)
            pred_keypoints3d = convert_keypoints(
                pred_keypoints3d,
                dst=self.eval_kps3d_convention,
                approximate=True)
            scene_gt_kps3d = gt_kps3d[start_idx:end_idx]
            gt_keypoints3d = Keypoints(
                kps=scene_gt_kps3d,
                mask=scene_gt_kps3d[..., -1:],
                convention=self.dataset.gt_kps3d_convention,
                logger=self.logger)
            gt_keypoints3d = convert_keypoints(
                gt_keypoints3d,
                dst=self.eval_kps3d_convention,
                approximate=True)
            eval_results = self.metric_manager(
                pred_keypoints3d=pred_keypoints3d,
                gt_keypoints3d=gt_keypoints3d)
            table = prettytable.PrettyTable()
            table.field_names = ['Metric name', 'Value']
            for metric_name, metric_dict in eval_results.items():
                for key, value in metric_dict.items():
                    table.add_row([f'{metric_name}_{key}', value])
            table_str = '\n' + table.get_string()
            self.logger.info(table_str)
            # save data and videos
            scene_dir = os.path.join(self.output_dir, f'scene_{scene_idx}')
            if not os.path.exists(scene_dir):
                os.mkdir(path=scene_dir)
            pred_kps3d_path = os.path.join(scene_dir, 'pred_keypoints3d.npz')
            pred_kps3d_paths.append(pred_kps3d_path)
            pred_keypoints3d.dump(pred_kps3d_path)
        if self.dataset_visualization is not None:
            self.dataset_visualization.pred_kps3d_paths = pred_kps3d_paths
            self.dataset_visualization.run(overwrite=overwrite)
