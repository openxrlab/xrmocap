# yapf: disable
import logging
import numpy as np
from prettytable import PrettyTable
from typing import List, Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.utils.geometry import compute_similarity_transform
from xrmocap.utils.mvpose_utils import (
    check_limb_is_correct, compute_mpjpe, vectorize_distance,
)

# yapf: enable


def evaluate(pred_keypoints3d: Keypoints,
             gt_keypoints3d: Keypoints,
             pck_thres: List = [50, 100],
             scale=1000.,
             logger: Union[None, str, logging.Logger] = None) -> dict:
    """evaluation of accuracy pred_keypoints3d (Keypoints):

    prediction of keypoints
    gt_keypoints3d (Keypoints):
        ground true of keypoints
    pck_thres (List):
        threshold value of precision
    scale:
    """
    # There must be no np.nan in the pred_keypoints3d
    mpjpe, pa_mpjpe = [], []
    pck = {i: [] for i in pck_thres}
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

            for thres in pck_thres:
                pck[thres].append(np.mean(pa_mpjpe_i <= (thres / scale)))

    mpjpe = np.asarray(mpjpe) * scale  # m to mm
    pa_mpjpe = np.asarray(pa_mpjpe) * scale  # m to mm
    mpjpe_mean, mpjpe_std = np.mean(mpjpe), np.std(mpjpe)
    pa_mpjpe_mean, pa_mpjpe_std = np.mean(pa_mpjpe), np.std(pa_mpjpe)
    # percentage
    for thres in pck_thres:
        pck[thres] = np.mean(pck[thres]) * 100.
    return dict(
        mpjpe_mean=mpjpe_mean,
        mpjpe_std=mpjpe_std,
        pa_mpjpe_mean=pa_mpjpe_mean,
        pa_mpjpe_std=pa_mpjpe_std,
        pck=pck)


def calc_limbs_accuracy(
    pred_keypoints3d,
    gt_keypoints3d,
    limbs,
    logger: Union[None, str, logging.Logger] = None
) -> Tuple[np.ndarray, PrettyTable]:
    """calculate the limbs accuracy pred_keypoints3d (Keypoints):

    prediction of keypoints
    gt_keypoints3d (Keypoints):
        ground true of keypoints
    limbs:
        limb to be evaluated
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
    bone_group = dict([('Torso', np.array([len(limbs) - 1])),
                       ('Upper arms', np.array([5, 6])),
                       ('Lower arms', np.array([4, 7])),
                       ('Upper legs', np.array([1, 2])),
                       ('Lower legs', np.array([0, 3]))])
    if len(limbs) > 9:
        # head is absent in some dataset
        bone_group['Head'] = np.array([8])

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
        f'Actor {i}' for i in range(bone_person_wise_result['Torso'].shape[0])
    ] + ['Average']
    for k, v in bone_person_wise_result.items():
        this_row = [k] + [np.char.mod('%.4f', i) for i in v
                          ] + [np.char.mod('%.4f',
                                           np.sum(v) / len(v))]
        tb.add_row(this_row)
    this_row = ['Total'] + [
        np.char.mod('%.4f', i) for i in person_wise_avg
    ] + [np.char.mod('%.4f',
                     np.sum(person_wise_avg) / len(person_wise_avg))]
    tb.add_row(this_row)
    return check_result, tb
