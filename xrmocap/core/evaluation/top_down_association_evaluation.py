# yapf: disable
import logging
import numpy as np
import os.path as osp
from prettytable import PrettyTable
from tqdm import tqdm
from typing import List, Union

from xrmocap.data.data_visualization.builder import BaseDataVisualization
from xrmocap.data.dataset.builder import MviewMpersonDataset
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.top_down_association.builder import (
    MvposeAssociator, build_top_down_associator,
)
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_idx, get_keypoint_num,
)
from xrmocap.utils.mvpose_utils import (
    add_campus_jaw_headtop, add_campus_jaw_headtop_mask,
)
from .base_evaluation import BaseEvaluation
from .metrics.base_metric import BaseMetric

# yapf: enable


class TopDownAssociationEvaluation(BaseEvaluation):
    """Top-down association evaluation."""

    def __init__(self,
                 dataset: Union[dict, MviewMpersonDataset],
                 output_dir: str,
                 metric_list: List[Union[dict, BaseMetric]],
                 associator: Union[dict, MvposeAssociator],
                 pick_dict: Union[dict, None] = None,
                 dataset_visualization: Union[None, dict,
                                              BaseDataVisualization] = None,
                 pred_kps3d_convention: str = 'coco',
                 eval_kps3d_convention: str = 'campus',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization for the class.

        Args:
            dataset (Union[dict, MviewMpersonDataset])
            output_dir (str): The path to save results.
            metric_list (List[Union[dict, BaseMetric]]):
                A list of metrics to be evaluated.
            associator (Union[dict, MvposeAssociator])
            pick_dict (Union[dict, None], optional):
                Selected metrics to be printed in the final table.
                Defaults to None.
            dataset_visualization
                (Union[None, dict, BaseDataVisualization], optional):
                Defaults to None.
            pred_kps3d_convention (str, optional): Target convention of
                keypoints3d, Defaults to 'coco'.
            eval_kps3d_convention (str, optional): the convention of
                keypoints3d for evaluation, Defaults to 'campus'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseEvaluation.__init__(
            self,
            dataset=dataset,
            output_dir=output_dir,
            metric_list=metric_list,
            pick_dict=pick_dict,
            dataset_visualization=dataset_visualization,
            eval_kps3d_convention=eval_kps3d_convention,
            logger=logger)

        self.pred_kps3d_convention = pred_kps3d_convention
        if isinstance(associator, dict):
            associator['logger'] = self.logger
            self.associator = build_top_down_associator(associator)
        else:
            self.associator = associator

    def run(self, overwrite: bool = False):
        # prepare output folder
        BaseEvaluation.run(self, overwrite=overwrite)

        n_frame = len(self.dataset)
        n_kps = get_keypoint_num(convention=self.pred_kps3d_convention)
        pred_kps3d = np.zeros(shape=(n_frame, 1, n_kps, 4))
        matched_kps2d_idx = [[] for _ in range(n_frame)]
        gt_kps3d = None
        max_identity = 0
        end_of_clip_idxs = []
        identities = []
        for frame_idx, frame_item in enumerate(tqdm(self.dataset)):
            mview_img_tensor, _, _, _, kps3d, end_of_clip, kw_data = frame_item
            if end_of_clip:
                end_of_clip_idxs.append(frame_idx)
            fisheye_list = self.dataset.fisheye_params[0]
            mview_img_arr = np.asarray(mview_img_tensor * 255).astype(np.uint8)
            mview_kps2d_list = kw_data['kps2d']
            mview_bbox2d_list = kw_data['bbox2d']
            # prepare input for associate single frame
            mview_keypoints2d_list = []
            for view_idx, kps2d in enumerate(mview_kps2d_list):
                keypoints2d = Keypoints(
                    dtype='numpy',
                    kps=kps2d.unsqueeze(0),
                    mask=kps2d[..., -1].unsqueeze(0) > 0,
                    convention=self.dataset.kps2d_convention,
                    logger=self.logger)
                mview_keypoints2d_list.append(keypoints2d)
            self.associator.set_cameras(fisheye_list)

            keypoints2d_idx, predict_keypoints3d, identities = \
                self.associator.associate_frame(
                    mview_img_arr=mview_img_arr,
                    mview_bbox2d=mview_bbox2d_list,
                    mview_keypoints2d=mview_keypoints2d_list,
                    affinity_type='geometry_mean'
                )

            # save predict kps3d
            for idx, identity in enumerate(identities):
                if identity > max_identity:
                    n_identity = identity - max_identity
                    pred_kps3d = np.concatenate(
                        (pred_kps3d,
                         np.zeros(shape=(n_frame, n_identity, n_kps, 4))),
                        axis=1)
                    max_identity = identity
                pred_kps3d[frame_idx,
                           identity] = predict_keypoints3d.get_keypoints()[0,
                                                                           idx]
            for i in range(max_identity + 1):
                if i in identities:
                    index = identities.index(i)
                    matched_kps2d_idx[frame_idx].append(keypoints2d_idx[index])
            # save ground truth kps3d
            if gt_kps3d is None:
                gt_kps3d = kps3d.numpy()[np.newaxis]
            else:
                gt_kps3d = np.concatenate(
                    (gt_kps3d, kps3d.numpy()[np.newaxis]), axis=0)

        pred_keypoints3d_raw = Keypoints(
            dtype='numpy',
            kps=pred_kps3d,
            mask=pred_kps3d[..., -1] > 0,
            convention=self.pred_kps3d_convention,
            logger=self.logger)
        gt_keypoints3d_raw = Keypoints(
            dtype='numpy',
            kps=gt_kps3d,
            mask=gt_kps3d[..., -1] > 0,
            convention=self.dataset.gt_kps3d_convention,
            logger=self.logger)

        mscene_keypoints_paths = []
        scene_start_idx = 0
        for scene_idx, scene_end_idx in enumerate(end_of_clip_idxs):
            scene_keypoints = pred_keypoints3d_raw.clone()
            kps3d = scene_keypoints.get_keypoints()[
                scene_start_idx:scene_end_idx + 1, ...]
            mask = scene_keypoints.get_mask()[scene_start_idx:scene_end_idx +
                                              1, ...]
            scene_keypoints.set_keypoints(kps3d)
            scene_keypoints.set_mask(mask)
            npz_path = osp.join(self.output_dir,
                                f'scene{scene_idx}_pred_keypoints3d.npz')
            scene_keypoints.dump(npz_path)
            scence_matched_kps2d_idx = matched_kps2d_idx[
                scene_start_idx:scene_end_idx + 1]
            np.save(
                osp.join(self.output_dir,
                         f'scene{scene_idx}_matched_kps2d_idx.npy'),
                scence_matched_kps2d_idx)
            mscene_keypoints_paths.append(npz_path)
            scene_start_idx = scene_end_idx + 1

        pred_keypoints3d, gt_keypoints3d = self.align_keypoints3d(
            pred_keypoints3d_raw, gt_keypoints3d_raw)
        # evaluate and print results
        eval_results, _ = self.metric_manager(
            pred_keypoints3d=pred_keypoints3d, gt_keypoints3d=gt_keypoints3d)
        table = PrettyTable()
        table.field_names = ['Metric name', 'Value']
        for metric_name, metric_dict in eval_results.items():
            for key, value in metric_dict.items():
                table.add_row([f'{metric_name}: {key}', f'{value:.2f}'])
        table_str = '\n' + table.get_string()
        self.logger.info(table_str)

        if self.dataset_visualization is not None:
            self.dataset_visualization.pred_kps3d_paths = \
                mscene_keypoints_paths
            self.dataset_visualization.run(overwrite=overwrite)

    def align_keypoints3d(self, pred_keypoints3d: Keypoints,
                          gt_keypoints3d: Keypoints):
        gt_nose = None
        pred_nose = None
        pred_kps3d_convention = pred_keypoints3d.get_convention()
        gt_kps3d_convention = gt_keypoints3d.get_convention()
        if gt_kps3d_convention == 'panoptic':
            gt_nose_index = get_keypoint_idx(
                name='nose_openpose', convention=gt_kps3d_convention)
            gt_nose = gt_keypoints3d.get_keypoints()[:, :, gt_nose_index, :3]

        if pred_kps3d_convention == 'coco':
            pred_nose_index = get_keypoint_idx(
                name='nose', convention=pred_kps3d_convention)
            pred_nose = pred_keypoints3d.get_keypoints()[:, :,
                                                         pred_nose_index, :3]

        if pred_kps3d_convention != self.eval_kps3d_convention:
            pred_keypoints3d = convert_keypoints(
                keypoints=pred_keypoints3d,
                dst=self.eval_kps3d_convention,
                approximate=True)
        if gt_kps3d_convention != self.eval_kps3d_convention:
            gt_keypoints3d = convert_keypoints(
                keypoints=gt_keypoints3d,
                dst=self.eval_kps3d_convention,
                approximate=True)

        pred_kps3d_mask = pred_keypoints3d.get_mask()
        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        if pred_nose is not None:
            pred_kps3d = add_campus_jaw_headtop(pred_nose, pred_kps3d)
            pred_kps3d_mask = add_campus_jaw_headtop_mask(pred_kps3d_mask)

        gt_kps3d_mask = gt_keypoints3d.get_mask()
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        if gt_nose is not None:
            gt_kps3d = add_campus_jaw_headtop(gt_nose, gt_kps3d)
            gt_kps3d_mask = add_campus_jaw_headtop_mask(gt_kps3d_mask)

        pred_kps3d = np.concatenate(
            (pred_kps3d, pred_kps3d_mask[..., np.newaxis]), axis=-1)
        pred_keypoints3d = Keypoints(
            kps=pred_kps3d,
            mask=pred_kps3d_mask,
            convention=self.eval_kps3d_convention)
        gt_kps3d = np.concatenate((gt_kps3d, gt_kps3d_mask[..., np.newaxis]),
                                  axis=-1)
        gt_keypoints3d = Keypoints(
            kps=gt_kps3d,
            mask=gt_kps3d_mask,
            convention=self.eval_kps3d_convention)

        return pred_keypoints3d, gt_keypoints3d
