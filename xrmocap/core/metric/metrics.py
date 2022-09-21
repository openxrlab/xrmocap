import csv
import logging
import numpy as np
import os.path as osp
import time
from prettytable import PrettyTable
from typing import Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.geometry import compute_similarity_transform
from xrmocap.utils.mvpose_utils import (
    check_limb_is_correct,
    compute_mpjpe,
    vectorize_distance,
)


def evaluate(pred_keypoints3d: Keypoints,
             gt_keypoints3d: Keypoints,
             pck_50_thres=50,
             pck_100_thres=100,
             scale=1000.,
             logger: Union[None, str, logging.Logger] = None) -> dict:
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
                    (f_gt_kps3d, np.ones_like(f_gt_kps3d[..., 0:1])), axis=-1),
                convention=gt_kps3d_convention)
            mpjpe.append(
                compute_mpjpe(f_pred_keypoints, f_gt_keypoints, align=True))

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
    logger.info(f'MPJPE: {mpjpe_mean:.2f} ± {mpjpe_std:.2f} mm')
    logger.info(f'PA-MPJPE: {pa_mpjpe_mean:.2f} ±' f'{pa_mpjpe_std:.2f} mm')
    logger.info(f'PCK@{pck_50_thres}mm: {pck_50:.2f} %')
    logger.info(f'PCK@{pck_100_thres}mm: {pck_100:.2f} %')
    return pck_50, pck_100, mpjpe, pa_mpjpe


def calc_limbs_accuracy(
    pred_keypoints3d,
    gt_keypoints3d,
    limbs,
    dump_dir=None,
    logger: Union[None, str,
                  logging.Logger] = None) -> Tuple[np.ndarray, list]:
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
        f'Actor {i}' for i in range(bone_person_wise_result['Head'].shape[0])
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
    ] + [np.char.mod('%.4f',
                     np.sum(person_wise_avg) / len(person_wise_avg))]
    tb.add_row(this_row)
    list_tb.append([
        float(i) if isinstance(i, type(np.array([]))) else i for i in this_row
    ])
    if dump_dir:
        np.save(
            osp.join(
                dump_dir,
                time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))),
            check_result)
        with open(
                osp.join(
                    dump_dir,
                    time.strftime('%Y_%m_%d_%H_%M.csv',
                                  time.localtime(time.time()))), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(list_tb)
    logger.info('\n' + tb.get_string())
    return check_result, list_tb
