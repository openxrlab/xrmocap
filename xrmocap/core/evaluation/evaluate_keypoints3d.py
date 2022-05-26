import argparse
import csv
import numpy as np
import os
import os.path as osp
import scipy.io as scio
import time
from collections import OrderedDict
from copy import deepcopy
from prettytable import PrettyTable
from typing import Tuple

from xrmocap.utils.geometry import compute_similarity_transform

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))


def convert_sense_omni_kp29_to_sense_omni_kp19(kp29: np.ndarray) -> np.ndarray:
    """transform coco order 29 keypoints 3d keypoints to coco order 19
    keypoints order with interpolation.

    Args:
        kp29 (np.ndarray): 3D keypoints with shape nx29x3

    Returns:
        np.ndarray: 3D keypoints with shape nx19x3
    """
    kp29 = np.asarray(kp29).copy()
    n, _, ndim = np.shape(kp29)
    rerank = [
        16, 20, 19, 24, 23, 12, 26, 25, 6, 8, 7, 14, 13, 18, 17, 0, 28, 27, 5
    ]
    kp19 = np.zeros((n, 19, ndim))
    kp19 = kp29[:, rerank]
    if ndim == 3:
        # headtop_y = nose_y - |head_y - nose_y|
        kp19[:, -1, 1] = kp19[:, 15, 1] - abs(kp19[:, -1, 1] - kp19[:, 15, 1])
    return kp19


def convert_sense_omni_kp19_to_shelf(coco_kps: np.ndarray) -> np.ndarray:
    """transform 19 keypoints order 3d keypoints to shelf dataset order with
    interpolation.

    Args:
        coco_kps (np.ndarray): 3D keypoints with shape 19x3

    Returns:
        np.ndarray: 3D keypoints in shelf order with shape 14x3
    """
    shelf_kps = np.zeros((14, 3))
    coco2shelf = np.array([7, 4, 2, 1, 3, 6, 14, 12, 10, 9, 11, 13])
    shelf_kps[0:12] += coco_kps[coco2shelf]
    shelf_kps[12] = (shelf_kps[8] +
                     shelf_kps[9]) / 2  # Use middle of shoulder to init
    shelf_kps[13] = coco_kps[15]  # use nose to init
    shelf_kps[13] = shelf_kps[12] + (shelf_kps[13] - shelf_kps[12]) * np.array(
        [0.75, 0.75, 1.5])
    shelf_kps[12] = shelf_kps[12] + (coco_kps[15] - shelf_kps[12]) * np.array(
        [1. / 2., 1. / 2., 1. / 2.])
    return shelf_kps


def convert_sense_omni_kp17_to_shelf(coco_kps: np.ndarray) -> np.ndarray:
    """transform coco order 3d keypoints to shelf dataset order with
    interpolation.

    Args:
        coco_kps (np.ndarray): 3D keypoints with shape 17x3

    Returns:
        np.ndarray: 3D keypoints in shelf order with shape 14x3
    """
    shelf_kps = np.zeros((14, 3))
    coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
    shelf_kps[0:12] += coco_kps[coco2shelf]
    shelf_kps[12] = (shelf_kps[8] +
                     shelf_kps[9]) / 2  # Use middle of shoulder to init
    shelf_kps[13] = coco_kps[0]  # use nose to init
    shelf_kps[13] = shelf_kps[12] + (shelf_kps[13] - shelf_kps[12]) * np.array(
        [0.75, 0.75, 1.5])
    shelf_kps[12] = shelf_kps[12] + (coco_kps[0] - shelf_kps[12]) * np.array(
        [1. / 2., 1. / 2., 1. / 2.])
    return shelf_kps


def convert_panoptic_to_sense_omni_kp19(pan_kps: np.ndarray) -> np.ndarray:
    """transform panoptic keypoints order to 19 keypoints order 3d keypoints.

    Args:
        pan_kps (np.ndarray): 3D keypoints in panoptic keypoints order
                               with shape 19x3.

    Returns:
        np.ndarray: 3D keypoints in coco keypoints order with shape 19x3.
    """
    omni_kps = np.zeros((19, 3))
    pan2omni = np.array(
        [2, 6, 12, 7, 13, -1, 8, 14, 0, 3, 9, 4, 10, 5, 11, 1, -1, -1, -1])
    omni_kps[pan2omni >= 0] = pan_kps[pan2omni[pan2omni >= 0]]

    return omni_kps


def vectorize_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate euclid distance on each row of a and b.

    Args:
        a (np.ndarray): Points in shape [N, ...]
        b (np.ndarray): Points in shape [M, ...]

    Returns:
        np.ndarray: Dist in shape [MxN, ...] representing correspond distance.
    """
    N = a.shape[0]
    a = a.reshape(N, -1)
    M = b.shape[0]
    b = b.reshape(M, -1)
    a2 = np.tile(np.sum(a**2, axis=1).reshape(-1, 1), (1, M))
    b2 = np.tile(np.sum(b**2, axis=1), (N, 1))
    dist = a2 + b2 - 2 * (a @ b.T)
    dist = np.sqrt(dist)
    dist[np.where(np.isnan(dist))] = 1000
    return dist


def check_bone_is_correct(model_start_point: np.ndarray,
                          model_end_point: np.ndarray,
                          gt_strat_point: np.ndarray,
                          gt_end_point: np.ndarray,
                          alpha=0.5) -> bool:
    """Check that bone predictions are correct.

    Returns:
        bool: If the predicted bone is correct, return True.
    """
    bone_lenth = np.linalg.norm(gt_end_point - gt_strat_point)
    start_difference = np.linalg.norm(gt_strat_point - model_start_point)
    end_difference = np.linalg.norm(gt_end_point - model_end_point)
    return ((start_difference + end_difference) / 2) <= alpha * bone_lenth


def align_by_root(keypoints):
    """Align keypoints (in shelf order) to their root (nose)"""
    root = keypoints[-1:, :]
    return keypoints - root


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, align=False):
    """Compute MPJPE given prediction and ground-truth.

    Args:
        pred (np.ndarray): points with shape Nx3,
            N means the number of keypoints.
        gt (np.ndarray): points with shape Nx3
        align (bool, optional): boolean value that determines whether to align
                                with the root. Defaults to False.

    Returns:
        MPJPE of the input keypoints
    """
    if align:
        pred = align_by_root(pred)
        gt = align_by_root(gt)
    mpjpe = np.sqrt(np.sum(np.square(pred - gt), axis=-1))
    return mpjpe


def evaluate(kps3d: np.ndarray,
             actor3D,
             range_,
             dataset_name='shelf_coco',
             dump_dir=None) -> Tuple[np.ndarray, list]:
    """Evaluation of skeletal accuracy with 14 keypoints.

    Args:
        kps3d (numpy.ndarray): 3d keypoints in every frame
        actor3D (numpy.ndarray): ground truth
        range_ (range): the range of frames
        dataset_name (str, optional): Defaults to 'shelf_coco'.
        dump_dir (str, optional): The path to save results. Defaults to None.
    """
    check_result = np.zeros((len(actor3D[0]), len(actor3D), 10),
                            dtype=np.int32)
    accuracy_cnt = 0
    error_cnt = 0
    for idx, img_id in enumerate(range_):
        for pid in range(len(actor3D)):
            if actor3D[pid][img_id][0].shape == (
                    1, 0) or actor3D[pid][img_id][0].shape == (0, 0):
                continue
            if 'coco' in dataset_name:
                model_kps3d = np.stack([
                    convert_sense_omni_kp17_to_shelf(i)
                    for i in deepcopy(kps3d[idx])
                ])
            elif 'kp29' in dataset_name:
                model_kps3d = convert_sense_omni_kp29_to_sense_omni_kp19(
                    deepcopy(kps3d[idx]))
                model_kps3d = np.stack([
                    convert_sense_omni_kp19_to_shelf(i)
                    for i in deepcopy(model_kps3d)
                ])
            else:
                model_kps3d = np.stack([
                    convert_sense_omni_kp19_to_shelf(i)
                    for i in deepcopy(kps3d[idx])
                ])
            gt_kp3d = actor3D[pid][img_id][0]
            if dataset_name[:8] == 'panoptic':
                gt_kp3d = convert_sense_omni_kp19_to_shelf(
                    convert_panoptic_to_sense_omni_kp19(gt_kp3d))
            dist = vectorize_distance(np.expand_dims(gt_kp3d, 0), model_kps3d)
            model_kp3d = model_kps3d[np.argmin(dist[0])]

            bones = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10],
                     [10, 11], [12, 13]]
            for i, bone in enumerate(bones):
                start_point, end_point = bone
                if check_bone_is_correct(model_kp3d[start_point],
                                         model_kp3d[end_point],
                                         gt_kp3d[start_point],
                                         gt_kp3d[end_point]):
                    check_result[img_id, pid, i] = 1
                    accuracy_cnt += 1
                else:
                    check_result[img_id, pid, i] = -1
                    error_cnt += 1
            gt_hip = (gt_kp3d[2] + gt_kp3d[3]) / 2
            model_hip = (model_kp3d[2] + model_kp3d[3]) / 2
            if check_bone_is_correct(model_hip, model_kp3d[12], gt_hip,
                                     gt_kp3d[12]):
                check_result[img_id, pid, -1] = 1
                accuracy_cnt += 1
            else:
                check_result[img_id, pid, -1] = -1
                error_cnt += 1

    bone_group = OrderedDict([('Head', np.array([8])),
                              ('Torso', np.array([9])),
                              ('Upper arms', np.array([5, 6])),
                              ('Lower arms', np.array([4, 7])),
                              ('Upper legs', np.array([1, 2])),
                              ('Lower legs', np.array([0, 3]))])

    person_wise_avg = np.sum(
        check_result > 0, axis=(0, 2)) / np.sum(
            np.abs(check_result), axis=(0, 2))

    bone_wise_result = OrderedDict()
    bone_person_wise_result = OrderedDict()
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
    print(tb)
    return check_result, list_tb


def evaluate_mpjpe(kps3d,
                   actor3D,
                   range_,
                   dump_dir=None,
                   pck_thres1=50,
                   pck_thres2=100,
                   dataset_name='shelf',
                   scale=1000.):
    zero_cnt = 0
    mpjpe, pa_mpjpe, pck1, pck2 = [], [], [], []

    for idx, img_id in enumerate(range_):
        for pid in range(len(actor3D)):

            if actor3D[pid][img_id][0].shape == (
                    1, 0) or actor3D[pid][img_id][0].shape == (0, 0):
                continue
            if 'coco' in dataset_name:
                model_kps3d = np.stack([
                    convert_sense_omni_kp17_to_shelf(i)
                    for i in deepcopy(kps3d[idx])
                ])
            elif 'kp29' in dataset_name:
                model_kps3d = convert_sense_omni_kp29_to_sense_omni_kp19(
                    deepcopy(kps3d[idx]))
                model_kps3d = np.stack([
                    convert_sense_omni_kp19_to_shelf(i)
                    for i in deepcopy(model_kps3d)
                ])
            else:
                model_kps3d = np.stack([
                    convert_sense_omni_kp19_to_shelf(i)
                    for i in deepcopy(kps3d[idx])
                ])
            gt_kp3d = actor3D[pid][img_id][0]
            if dataset_name[:8] == 'panoptic':
                gt_kp3d = convert_sense_omni_kp19_to_shelf(
                    convert_panoptic_to_sense_omni_kp19(gt_kp3d))
            dist = vectorize_distance(np.expand_dims(gt_kp3d, 0), model_kps3d)
            model_kp3d = model_kps3d[np.argmin(dist[0])]

            model_kp3d[np.where(np.isnan(model_kp3d))] = 0.101

            if np.all((model_kp3d == 0)):
                zero_cnt += 1
                print('v distance all zero(count, fram_id, total_frame)',
                      zero_cnt, idx, len(range_))
                continue

            # MPJPE
            _mpjpe = compute_mpjpe(model_kp3d, gt_kp3d, align=True)
            mpjpe.append(_mpjpe)

            # PA-MPJPE
            _, Z, T, b, c = compute_similarity_transform(
                gt_kp3d, model_kp3d, compute_optimal_scale=True)
            model_kp_pa = (b * model_kp3d.dot(T)) + c
            _pa_mpjpe = compute_mpjpe(model_kp_pa, gt_kp3d, align=True)
            pa_mpjpe.append(_pa_mpjpe)

            _pck1 = np.mean(_pa_mpjpe <= (pck_thres1 / scale))
            _pck2 = np.mean(_pa_mpjpe <= (pck_thres2 / scale))
            pck1.append(_pck1)
            pck2.append(_pck2)

    mpjpe = np.asarray(mpjpe) * scale  # m to mm
    pa_mpjpe = np.asarray(pa_mpjpe) * scale  # m to mm
    if 'panoptic' in dataset_name:
        mpjpe = mpjpe.mean(axis=0)
        pa_mpjpe = pa_mpjpe.mean(axis=0)
    mpjpe_mean, mpjpe_std = np.mean(mpjpe), np.std(mpjpe)
    pa_mpjpe_mean, pa_mpjpe_std = np.mean(pa_mpjpe), np.std(pa_mpjpe)
    pck1 = np.mean(pck1) * 100.  # percentage
    pck2 = np.mean(pck2) * 100.  # percentage
    print(f'- MPJPE: {mpjpe_mean:.2f} ± {mpjpe_std:.2f} mm')
    print(f'- PA-MPJPE: {pa_mpjpe_mean:.2f} ± {pa_mpjpe_std:.2f} mm')
    print(f'- PCK@{pck_thres1}mm: {pck1:.2f} %')
    print(f'- PCK@{pck_thres2}mm: {pck2:.2f} %')

    return mpjpe, pa_mpjpe, pck1, pck2, [[mpjpe_mean, mpjpe_std],
                                         [pa_mpjpe_mean, pa_mpjpe_std],
                                         [pck1, pck2]]


if __name__ == '__main__':
    # np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', dest='datasets', required=True)
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--input_path', '-i', default='data')
    parser.add_argument('--result_path', '-r', default='data')
    parser.add_argument('--start_frame', '-s', type=int, default=300)
    parser.add_argument('--end_frame', '-e', type=int, default=600)
    args = parser.parse_args()

    for _, dataset_name in enumerate(args.datasets):
        data_name = dataset_name.split('_')[0]
        if data_name == 'shelf':
            gt_path = osp.join(args.input_path, data_name)
            test_range = range(args.start_frame, args.end_frame)
            keypoints = np.load(
                osp.join(
                    args.result_path, data_name,
                    f'{args.start_frame}_{args.end_frame-1}_human.pickle'),
                allow_pickle=True)
            save_dir = osp.join(args.result_path, data_name, args.exp_name)
            os.makedirs(save_dir, exist_ok=True)

        elif data_name == 'campus':
            gt_path = osp.join(args.input_path, data_name)
            test_range = [i for i in range(args.start_frame, args.end_frame)
                          ] + [i for i in range(650, 750)]
            # test_range = [i for i in range(650, 750)]
            keypoints1 = np.load(
                osp.join(
                    args.result_path, data_name,
                    f'{args.start_frame}_{args.end_frame-1}_human.pickle'),
                allow_pickle=True)
            keypoints2 = np.load(
                osp.join(args.result_path, data_name, '650_749_human.pickle'),
                allow_pickle=True)
            if keypoints1.shape[1] != keypoints2.shape[1]:
                cnt = abs(keypoints1.shape[1] - keypoints2.shape[1])
                if keypoints1.shape[1] < keypoints2.shape[1]:
                    zeros = np.zeros(
                        (keypoints1.shape[0], 1, keypoints1.shape[2],
                         keypoints1.shape[3])).repeat(
                             cnt, axis=1)
                    keypoints1 = np.concatenate([keypoints1, zeros], axis=1)
                else:
                    zeros = np.zeros(
                        (keypoints2.shape[0], 1, keypoints2.shape[2],
                         keypoints2.shape[3])).repeat(
                             cnt, axis=1)
                    keypoints2 = np.concatenate([keypoints2, zeros], axis=1)
            keypoints = np.concatenate([keypoints1, keypoints2], axis=0)
            save_dir = osp.join(args.result_path, data_name, args.exp_name)
            os.makedirs(save_dir, exist_ok=True)

        elif data_name == 'panoptic':
            name = dataset_name.split('_')[1]
            gt_path = os.path.join(args.input_path, 'panoptic',
                                   f'panoptic_{name}')
            test_range = range(args.start_frame, args.end_frame)
            keypoints = np.load(
                osp.join(
                    args.result_path, f'panoptic_{name}',
                    f'{args.start_frame}_{args.end_frame-1}_human.pickle'),
                allow_pickle=True)
            save_dir = osp.join(args.result_path, f'panoptic_{name}',
                                args.exp_name)
            os.makedirs(save_dir, exist_ok=True)

        else:
            NotImplementedError

        if data_name == 'shelf' or data_name == 'campus':
            actorsGT = scio.loadmat(osp.join(gt_path, 'actorsGT.mat'))
            gt3d = actorsGT['actor3D'][0]
            gt3d = gt3d[:3]
            evaluate(
                keypoints, gt3d, test_range, dataset_name, dump_dir=save_dir)
            # other metrics
            evaluate_mpjpe(
                keypoints, gt3d, test_range, dataset_name=dataset_name)
        elif data_name == 'panoptic':
            actorsGT = scio.loadmat(osp.join(gt_path, 'actorsGT.mat'))
            gt3d = actorsGT['actor3D'][0]
            gt3d = gt3d[:4]
            evaluate(
                keypoints, gt3d, test_range, dataset_name, dump_dir=save_dir)
            evaluate_mpjpe(
                keypoints, gt3d, test_range, dataset_name=dataset_name)

        else:
            NotImplementedError
