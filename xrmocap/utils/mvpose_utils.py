import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.limbs import get_limbs_from_keypoints


def get_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    """Get 2d-2d point distance, return error if any of the input is nan.

    Args:
        x (np.ndarray): 2d point
        y (np.ndarray): 2d point

    Returns:
        np.float64: distance between two 2d points
    """
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    else:
        return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5


def projected_distance(pts_0,
                       pts_1,
                       F,
                       scores_0=None,
                       scores_1=None,
                       n_kps2d=17):
    """Compute point distance with epipolar geometry knowledge.

    :param pts_0: numpy points array with shape Nx17x2
    :param pts_1: numpy points array with shape Nx17x2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """

    lines = cv2.computeCorrespondEpilines(pts_0.reshape(-1, 1, 2), 2, F)
    lines = lines.reshape(-1, n_kps2d, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], n_kps2d, 3])
    points_1[0, :, :, :2] = pts_1

    dist = np.sum(lines * points_1, axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)

    return dist


def geometry_affinity(points_set: np.ndarray, Fs: np.ndarray,
                      dim_group: torch.Tensor, factor=5, n_kps2d=17) \
                      -> np.ndarray:
    """Get geometry affinity.

    Args:
        points_set (np.ndarray): Keypoints of the model, in shape
            (total number of people detected, n_kps2d, 2)
        Fs (np.ndarray): (num_cam, num_cam, 3, 3)
        dim_group (torch.Tensor): The cumulative number of people from
            different perspectives. e.g. If you have five cameras and
            each camera detects two people, dim_group is
            tensor([ 0,  2,  4,  6,  8, 10]).
        factor (int, optional): Defaults to 5.
        n_kps2d (int, optional): the number of keypoints2d. Defaults to 17.

    Returns:
        affinity_matrix (np.ndarray): affinity matrix in shape (N, N),
            N = n1+n2+..., n1 is the number of detected people in cam1
    """
    M, _, _ = points_set.shape
    dist = np.ones((M, M), dtype=np.float32) * factor**2
    np.fill_diagonal(dist, 0)
    # TODO: remove this stupid nested for loop
    for cam_id0, h in enumerate(range(len(dim_group) - 1)):
        for cam_add, k in enumerate(range(cam_id0 + 1, len(dim_group) - 1)):
            cam_id1 = cam_id0 + cam_add + 1
            # if there is no one in some view, skip it!
            if dim_group[h] == dim_group[h +
                                         1] or dim_group[k] == dim_group[k +
                                                                         1]:
                continue

            kps2d_id0 = points_set[dim_group[h]:dim_group[h + 1]]
            kps2d_id1 = points_set[dim_group[k]:dim_group[k + 1]]
            dist[dim_group[h]:dim_group[h+1], dim_group[k]:dim_group[k+1]]\
                = projected_distance(kps2d_id0, kps2d_id1,
                                     Fs[cam_id0, cam_id1],
                                     n_kps2d=n_kps2d)/2\
                + projected_distance(kps2d_id1, kps2d_id0,
                                     Fs[cam_id1, cam_id0],
                                     n_kps2d=n_kps2d).T/2
            dist[dim_group[k]:dim_group[k+1], dim_group[h]:dim_group[h+1]] =\
                dist[dim_group[h]:dim_group[h+1],
                     dim_group[k]:dim_group[k+1]].T
    if dist.std() < factor:
        for i in range(dist.shape[0]):
            dist[i, i] = dist.mean()

    affinity_matrix = -(dist - dist.mean()) / (dist.std() + 1e-12)
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp(-factor * affinity_matrix))
    return affinity_matrix


def check_bone_length(kps3d: np.ndarray, convention: str = 'coco') -> bool:
    """Check selected bone length.

    Args:
        kps3d (np.ndarray): 3xN 3D keypoints in MSCOCO order.
        convention (str, optional): Keypoints factory.. Defaults to 'coco'.

    Raises:
        NotImplementedError: The type of connection is not available。

    Returns:
        bool: If true, the selected bone length is satisfaction
    """
    min_length = 0.1
    max_length = 0.7
    kps3d = kps3d.transpose(1, 0)
    kps3d_score = np.ones((kps3d.shape[0], 1))
    kps3d = np.concatenate((kps3d, kps3d_score), axis=1)
    keypoints3d = Keypoints(kps=kps3d, convention=convention)
    limbs = get_limbs_from_keypoints(
        keypoints=keypoints3d, fill_limb_names=True)
    all_conn_dict = limbs.get_connections_by_names()
    selected_conn = []

    if convention == 'coco':
        selected_conn_name = [
            'left_lower_leg', 'left_thigh', 'right_lower_leg', 'right_thigh',
            'left_upperarm', 'right_upperarm', 'left_forearm', 'right_forearm'
        ]
        for key, value in all_conn_dict.items():
            if key in selected_conn_name:
                selected_conn.append(value)
    else:
        raise NotImplementedError('Other conventions are not yet implemented.')
    error_cnt = 0
    for kp_0, kp_1 in selected_conn:
        conn_length = np.sqrt(np.sum((kps3d[kp_0, :3] - kps3d[kp_1, :3])**2))
        if conn_length < min_length or conn_length > max_length:
            error_cnt += 1
    return error_cnt < 3


def visualize_match(frame_id,
                    n_camera,
                    matched_list,
                    sub_imgid2cam,
                    img_bboxes,
                    track_id,
                    input_folder,
                    img_folder,
                    cam_offset=4,
                    save_folder='./result'):
    data_name = input_folder.split('/')[-1]
    cols = len(matched_list) + 1
    rows = sub_imgid2cam.max() + 2
    cam_offset = 0
    if data_name == 'campus':
        imgs = [
            cv2.imread(f'{img_folder}/{data_name}/img/' +
                       f'Camera{cam_id}/campus4-c{cam_id}-{frame_id:05d}.png')
            for cam_id in range(cam_offset, cam_offset + n_camera)
        ]
    elif 'panoptic' in data_name:
        imgs = [
            cv2.imread(f'{img_folder}/{data_name}/img/' +
                       f'Camera{cam_id}/frame_{frame_id:06d}.png')
            for cam_id in range(cam_offset, cam_offset + n_camera)
        ]
    elif data_name == 'shelf':
        imgs = [
            cv2.imread(f'{img_folder}/{data_name}/img/' +
                       f'Camera{cam_id}/img_{frame_id:06d}.png')
            for cam_id in range(cam_offset, cam_offset + n_camera)
        ]
    else:
        NotImplementedError
    for i, person in enumerate(matched_list):
        # Plot origin image
        for sub_imageid in person:
            cam_id = sub_imgid2cam[sub_imageid]
            bbox = img_bboxes[sub_imageid]
            bbox[bbox < 0] = 0
            pid = track_id[sub_imageid]
            cropped_img = imgs[cam_id][int(bbox[1]):int(bbox[3]),
                                       int(bbox[0]):int(bbox[2])]

            plt.subplot(rows, cols, cam_id * cols + i + 2)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            plt.imshow(cropped_img)
            plt.xlabel(f'#{sub_imageid}@{pid}')
            plt.xticks([])
            plt.yticks([])

    os.makedirs(f'{save_folder}', exist_ok=True)
    plt.savefig(f'{save_folder}/match_{int(frame_id)}.png')
    plt.close()


def get_min_reprojection_error(person, mview_proj_mat, kps2d_mat,
                               sub_imgid2cam):
    reproj_error = np.zeros((len(person), len(person)))
    for i, p0 in enumerate(person):
        for j, p1 in enumerate(person):
            projmat_0 = mview_proj_mat[sub_imgid2cam[p0]]
            projmat_1 = mview_proj_mat[sub_imgid2cam[p1]]
            kps2d_0, kps2d_1 = kps2d_mat[p0].T, kps2d_mat[p1].T
            kps3d_homo = cv2.triangulatePoints(projmat_0, projmat_1, kps2d_0,
                                               kps2d_1)
            this_error = 0
            for pk in person:
                projmat_k = mview_proj_mat[sub_imgid2cam[pk]]
                projected_kps_k_homo = projmat_k @ kps3d_homo
                projected_kps_k = \
                    projected_kps_k_homo[:2] / projected_kps_k_homo[2]
                this_error += np.linalg.norm(projected_kps_k - kps2d_mat[pk].T)
            reproj_error[i, j] = this_error

    reproj_error[np.arange(len(person)), np.arange(len(person))] = np.inf
    # TODO: figure out why NaN
    reproj_error[np.isnan(reproj_error)] = np.inf
    x, y = np.where(reproj_error == reproj_error.min())
    sub_imageid = np.array([person[x[0]], person[y[0]]])
    return sub_imageid
