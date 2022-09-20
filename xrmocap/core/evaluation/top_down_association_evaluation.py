# yapf: disable
import csv
import logging
import numpy as np
import os.path as osp
import time
from prettytable import PrettyTable
from tqdm import tqdm
from typing import List, Tuple, Union
from xrprimer.utils.log_utils import get_logger
from xrprimer.utils.path_utils import prepare_output_path

from xrmocap.data.data_visualization.builder import (
    BaseDataVisualization, build_data_visualization,
)
from xrmocap.data.dataset.builder import MviewMpersonDataset, build_dataset
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.ops.top_down_association.builder import (
    MvposeAssociator, build_top_down_associator,
)
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_idx, get_keypoint_num,
)
from xrmocap.transform.limbs import get_limbs_from_keypoints
from xrmocap.utils.geometry import compute_similarity_transform
from xrmocap.utils.mvpose_utils import (
    add_campus_jaw_headtop, add_campus_jaw_headtop_mask, check_limb_is_correct,
    compute_mpjpe, vectorize_distance,
)

# yapf: enable


class TopDownAssociationEvaluation:
    """Top-down association evaluation."""

    def __init__(self,
                 output_dir: str,
                 selected_limbs_name: List[List[str]],
                 additional_limbs_names: List[List[str]],
                 dataset: Union[dict, MviewMpersonDataset],
                 associator: Union[dict, MvposeAssociator],
                 dataset_visualization: Union[None, dict,
                                              BaseDataVisualization] = None,
                 pred_kps3d_convention: str = 'coco',
                 eval_kps3d_convention: str = 'campus',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization for the class.

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
            self.associator = build_top_down_associator(associator)
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
            scence_matched_kps2d_idx = matched_kps2d_idx[
                scene_start_idx:scene_end_idx + 1]
            np.save(
                osp.join(self.output_dir,
                         f'scene{scene_idx}_matched_kps2d_idx.npy'),
                scence_matched_kps2d_idx)
            mscene_keypoints_paths.append(npz_path)
            scene_start_idx = scene_end_idx + 1

        pred_keypoints3d_, gt_keypoints3d_, limbs = self.align_keypoints3d(
            pred_keypoints3d, gt_keypoints3d)
        self.calc_limbs_accuracy(pred_keypoints3d_, gt_keypoints3d_, limbs)
        pck_50, pck_100, mpjpe, pa_mpjpe = self.evaluate(
            pred_keypoints3d_, gt_keypoints3d_)

        if self.dataset_visualization is not None:
            self.dataset_visualization.pred_kps3d_paths = \
                mscene_keypoints_paths
            self.dataset_visualization.run(overwrite=overwrite)

    def evaluate(self,
                 pred_keypoints3d: Keypoints,
                 gt_keypoints3d: Keypoints,
                 pck_50_thres=50,
                 pck_100_thres=100,
                 scale=1000.) -> dict:
        # There must be no np.nan in the pred_keypoints3d
        mpjpe, pa_mpjpe, pck_50, pck_100 = [], [], [], []
        n_frame = gt_keypoints3d.get_frame_number()
        gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
        gt_kps3d_mask = gt_keypoints3d.get_mask()
        pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
        pred_kps3d_mask = pred_keypoints3d.get_mask()
        pred_kps3d_convention = pred_keypoints3d.get_convention()
        gt_kps3d_convention = gt_keypoints3d.get_convention()
        for frame_idx in range(n_frame):
            if not gt_kps3d_mask[frame_idx].any():
                continue
            gt_kps3d_idxs = np.where(
                np.sum(gt_kps3d_mask[frame_idx], axis=1) > 0)[0]
            for gt_kps3d_idx in gt_kps3d_idxs:
                f_gt_kps3d = gt_kps3d[frame_idx][gt_kps3d_idx]
                f_pred_kps3d = pred_kps3d[frame_idx][
                    np.sum(pred_kps3d_mask[frame_idx], axis=1) > 0]
                if len(f_pred_kps3d) == 0:
                    continue

                dist = vectorize_distance(f_gt_kps3d[np.newaxis], f_pred_kps3d)
                f_pred_kps3d = f_pred_kps3d[np.argmin(dist[0])]

                if np.all((f_pred_kps3d == 0)):
                    continue

                # MPJPE
                f_pred_keypoints = Keypoints(
                    kps=np.concatenate(
                        (f_pred_kps3d, np.ones_like(f_pred_kps3d[..., 0:1])),
                        axis=-1),
                    convention=pred_kps3d_convention)
                f_gt_keypoints = Keypoints(
                    kps=np.concatenate(
                        (f_gt_kps3d, np.ones_like(f_gt_kps3d[..., 0:1])),
                        axis=-1),
                    convention=gt_kps3d_convention)
                mpjpe.append(
                    compute_mpjpe(
                        f_pred_keypoints, f_gt_keypoints, align=True))

                # PA-MPJPE
                _, _, rotation, scaling, transl = compute_similarity_transform(
                    f_gt_kps3d, f_pred_kps3d, compute_optimal_scale=True)
                pred_kps3d_pa = (scaling * f_pred_kps3d.dot(rotation)) + transl

                pred_keypoints_pa = Keypoints(
                    kps=np.concatenate(
                        (pred_kps3d_pa, np.ones_like(pred_kps3d_pa[..., 0:1])),
                        axis=-1),
                    convention=pred_kps3d_convention)
                pa_mpjpe_i = compute_mpjpe(
                    pred_keypoints_pa, f_gt_keypoints, align=True)
                pa_mpjpe.append(pa_mpjpe_i)

                pck_50.append(np.mean(pa_mpjpe_i <= (pck_50_thres / scale)))
                pck_100.append(np.mean(pa_mpjpe_i <= (pck_100_thres / scale)))
        mpjpe = np.asarray(mpjpe) * scale  # m to mm
        pa_mpjpe = np.asarray(pa_mpjpe) * scale  # m to mm
        mpjpe_mean, mpjpe_std = np.mean(mpjpe), np.std(mpjpe)
        pa_mpjpe_mean, pa_mpjpe_std = np.mean(pa_mpjpe), np.std(pa_mpjpe)
        pck_50 = np.mean(pck_50) * 100.  # percentage
        pck_100 = np.mean(pck_100) * 100.  # percentage
        self.logger.info(f'MPJPE: {mpjpe_mean:.2f} ± {mpjpe_std:.2f} mm')
        self.logger.info(f'PA-MPJPE: {pa_mpjpe_mean:.2f} ±'
                         f'{pa_mpjpe_std:.2f} mm')
        self.logger.info(f'PCK@{pck_50_thres}mm: {pck_50:.2f} %')
        self.logger.info(f'PCK@{pck_100_thres}mm: {pck_100:.2f} %')
        return pck_50, pck_100, mpjpe, pa_mpjpe

    def align_keypoints3d(self, pred_keypoints3d: Keypoints,
                          gt_keypoints3d: Keypoints):
        ret_limbs = []
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

        limbs = get_limbs_from_keypoints(
            keypoints=pred_keypoints3d, fill_limb_names=True)
        limb_name_list = []
        conn_list = []
        for limb_name, conn in limbs.get_connections_by_names().items():
            limb_name_list.append(limb_name)
            conn_list.append(conn)

        for idx, limb_name in enumerate(limb_name_list):
            if limb_name in self.selected_limbs_name:
                ret_limbs.append(conn_list[idx])

        for conn_names in self.additional_limbs_names:
            kps_idx_0 = get_keypoint_idx(
                name=conn_names[0], convention=self.eval_kps3d_convention)
            kps_idx_1 = get_keypoint_idx(
                name=conn_names[1], convention=self.eval_kps3d_convention)
            ret_limbs.append(np.array([kps_idx_0, kps_idx_1], dtype=np.int32))
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

        return pred_keypoints3d, gt_keypoints3d, ret_limbs

    def calc_limbs_accuracy(self,
                            pred_keypoints3d,
                            gt_keypoints3d,
                            limbs,
                            dump_dir=None) -> Tuple[np.ndarray, list]:
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
                                             f_gt_kps3d[end_point]):
                        check_result[idx, gt_kps3d_idx, i] = 1
                        accuracy_cnt += 1
                    else:
                        check_result[idx, gt_kps3d_idx, i] = -1
                        error_cnt += 1
                gt_hip = (f_gt_kps3d[2] + f_gt_kps3d[3]) / 2
                pred_hip = (f_pred_kps3d[2] + f_pred_kps3d[3]) / 2
                if check_limb_is_correct(pred_hip, f_pred_kps3d[12], gt_hip,
                                         f_gt_kps3d[12]):
                    check_result[idx, gt_kps3d_idx, -1] = 1
                    accuracy_cnt += 1
                else:
                    check_result[idx, gt_kps3d_idx, -1] = -1
                    error_cnt += 1
        bone_group = dict([('Head', np.array([8])), ('Torso', np.array([9])),
                           ('Upper arms', np.array([5, 6])),
                           ('Lower arms', np.array([4, 7])),
                           ('Upper legs', np.array([1, 2])),
                           ('Lower legs', np.array([0, 3]))])

        person_wise_avg = np.sum(
            check_result > 0, axis=(0, 2)) / np.sum(
                np.abs(check_result), axis=(0, 2))

        bone_wise_result = dict()
        bone_person_wise_result = dict()
        for k, v in bone_group.items():
            bone_wise_result[k] = np.sum(check_result[:, :, v] > 0) / np.sum(
                np.abs(check_result[:, :, v]))
            bone_person_wise_result[k] = np.sum(
                check_result[:, :, v] > 0, axis=(0, 2)) / np.sum(
                    np.abs(check_result[:, :, v]), axis=(0, 2))

        tb = PrettyTable()
        tb.field_names = ['Bone Group'] + [
            f'Actor {i}'
            for i in range(bone_person_wise_result['Head'].shape[0])
        ] + ['Average']
        list_tb = [tb.field_names]
        for k, v in bone_person_wise_result.items():
            this_row = [k] + [np.char.mod('%.4f', i) for i in v
                              ] + [np.char.mod('%.4f',
                                               np.sum(v) / len(v))]
            list_tb.append([
                float(i) if isinstance(i, type(np.array([]))) else i
                for i in this_row
            ])
            tb.add_row(this_row)
        this_row = ['Total'] + [
            np.char.mod('%.4f', i) for i in person_wise_avg
        ] + [
            np.char.mod('%.4f',
                        np.sum(person_wise_avg) / len(person_wise_avg))
        ]
        tb.add_row(this_row)
        list_tb.append([
            float(i) if isinstance(i, type(np.array([]))) else i
            for i in this_row
        ])
        if dump_dir:
            np.save(
                osp.join(
                    dump_dir,
                    time.strftime('%Y_%m_%d_%H_%M',
                                  time.localtime(time.time()))), check_result)
            with open(
                    osp.join(
                        dump_dir,
                        time.strftime('%Y_%m_%d_%H_%M.csv',
                                      time.localtime(time.time()))), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(list_tb)
        self.logger.info('\n' + tb.get_string())
        return check_result, list_tb
