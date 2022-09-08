# yapf: disable
import logging
import numpy as np
import os.path as osp
from tqdm import tqdm
from typing import List, Union
from xrprimer.utils.log_utils import get_logger
from xrprimer.utils.path_utils import prepare_output_path

from xrmocap.data.data_visualization.builder import (
    BaseDataVisualization, build_data_visualization,
)
from xrmocap.data.dataset.builder import MviewMpersonDataset, build_dataset
from xrmocap.data_structure.keypoints import Keypoints

from xrmocap.ops.bottom_up_association.builder import (
    FourDAGAssociator, build_bottom_up_associator
)
from xrmocap.transform.convention.keypoints_convention import get_keypoint_num
from xrmocap.core.metric.metrics import (
    evaluate,calc_limbs_accuracy
)
from xrmocap.core.alignment.align_keypoints3d import align_keypoints3d
# yapf: enable

class BottomUpAssociationEvaluation:

    def __init__(self,
                 output_dir: str,
                 selected_limbs_name: List[List[str]],
                 additional_limbs_names: List[List[str]],
                 dataset: Union[dict, MviewMpersonDataset],
                 associator: Union[dict, FourDAGAssociator],
                 dataset_visualization: Union[None, dict,
                                              BaseDataVisualization] = None,
                 pred_kps3d_convention: str = 'coco',
                 eval_kps3d_convention: str = 'campus',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Top-down association evaluation.

        Args:
            output_dir (str): The path to save results.
            selected_limbs_name (List[List[str]]): The name of selected
                limbs in evaluation.
            additional_limbs_names (List[List[str]]):
                Names at both ends of the limb.
            dataset (Union[dict, MviewMpersonDataset])
            associator (Union[dict, MvposeAssociator])
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

        self.output_dir = output_dir
        self.pred_kps3d_convention = pred_kps3d_convention
        self.eval_kps3d_convention = eval_kps3d_convention
        self.additional_limbs_names = additional_limbs_names
        self.selected_limbs_name = selected_limbs_name
        self.logger = get_logger(logger)

        if isinstance(dataset, dict):
            dataset['logger'] = self.logger
            self.dataset = build_dataset(dataset)
        else:
            self.dataset = dataset

        if isinstance(associator, dict):
            associator['logger'] = self.logger
            self.associator = build_bottom_up_associator(associator)
        else:
            self.associator = associator

        if isinstance(dataset_visualization, dict):
            dataset_visualization['logger'] = self.logger
            self.dataset_visualization = build_data_visualization(
                dataset_visualization)
        else:
            self.dataset_visualization = dataset_visualization

    def run(self, overwrite: bool = False):
        prepare_output_path(
            output_path=self.output_dir,
            allowed_suffix='',
            path_type='dir',
            overwrite=overwrite,
            logger=self.logger)
        n_frame = len(self.dataset)
        n_kps = get_keypoint_num(convention=self.pred_kps3d_convention)
        pred_kps3d = np.zeros(shape=(n_frame, 1, n_kps, 4))
        gt_kps3d = None
        max_identity = 0
        end_of_clip_idxs = []
        identities = []
        for frame_idx, frame_item in enumerate(tqdm(self.dataset)):
            _, _, _, _, kps3d, end_of_clip, kw_data = frame_item
            if end_of_clip:
                end_of_clip_idxs.append(frame_idx)
            fisheye_list = self.dataset.fisheye_params[0]
            # prepare input for associate single frame
            
            self.associator.set_cameras(fisheye_list)

            predict_keypoints3d, identities = \
                self.associator.associate_frame(kw_data)
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
            # save ground truth kps3d
            if gt_kps3d is None:
                gt_kps3d = kps3d.numpy()[np.newaxis]
            else:
                gt_kps3d = np.concatenate(
                    (gt_kps3d, kps3d.numpy()[np.newaxis]), axis=0)

        pred_keypoints3d = Keypoints(
            dtype='numpy',
            kps=pred_kps3d,
            mask=pred_kps3d[..., -1] > 0,
            convention=self.pred_kps3d_convention,
            logger=self.logger)
        gt_keypoints3d = Keypoints(
            dtype='numpy',
            kps=gt_kps3d,
            mask=gt_kps3d[..., -1] > 0,
            convention=self.dataset.gt_kps3d_convention,
            logger=self.logger)

        mscene_keypoints_paths = []
        scene_start_idx = 0
        for scene_idx, scene_end_idx in enumerate(end_of_clip_idxs):
            scene_keypoints = pred_keypoints3d.clone()
            kps3d = scene_keypoints.get_keypoints()[
                scene_start_idx:scene_end_idx + 1, ...]
            mask = scene_keypoints.get_mask()[scene_start_idx:scene_end_idx +
                                              1, ...]
            scene_keypoints.set_keypoints(kps3d)
            scene_keypoints.set_mask(mask)
            npz_path = osp.join(self.output_dir,
                                f'scene{scene_idx}_pred_keypoints3d.npz')
            scene_keypoints.dump(npz_path)
            mscene_keypoints_paths.append(npz_path)
            scene_start_idx = scene_end_idx + 1

        pred_keypoints3d_, gt_keypoints3d_, limbs = align_keypoints3d(
            pred_keypoints3d, gt_keypoints3d,self.eval_kps3d_convention,self.selected_limbs_name,self.additional_limbs_names)
        calc_limbs_accuracy(pred_keypoints3d_, gt_keypoints3d_, limbs,logger=self.logger)
        pck_50, pck_100, mpjpe, pa_mpjpe = evaluate(
            pred_keypoints3d_, gt_keypoints3d_,logger=self.logger)

        if self.dataset_visualization is not None:
            self.dataset_visualization.pred_kps3d_paths = \
                mscene_keypoints_paths
            self.dataset_visualization.run(overwrite=overwrite)
