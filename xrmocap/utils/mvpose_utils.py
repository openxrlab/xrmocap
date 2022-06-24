import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.limbs import get_limbs_from_keypoints
from .ffmpeg_utils import array_to_video

_ALL_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
              (0, 255, 255), (255, 255, 0), (215, 255, 0), (215, 0, 255),
              (0, 215, 255), (255, 215, 0), (255, 0, 215), (255, 0, 215),
              (0, 155, 255), (155, 0, 255), (155, 255, 0)]


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
        raise NotImplementedError
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


def get_min_reprojection_error(person, dataset, kps2d_mat, sub_imgid2cam):
    reproj_error = np.zeros((len(person), len(person)))
    for i, p0 in enumerate(person):
        for j, p1 in enumerate(person):
            projmat_0 = dataset.P[sub_imgid2cam[p0]]
            projmat_1 = dataset.P[sub_imgid2cam[p1]]
            kps2d_0, kps2d_1 = kps2d_mat[p0].T, kps2d_mat[p1].T
            kps3d_homo = cv2.triangulatePoints(projmat_0, projmat_1, kps2d_0,
                                               kps2d_1)
            this_error = 0
            for pk in person:
                projmat_k = dataset.P[sub_imgid2cam[pk]]
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


def draw_skeleton(aa, kp, show_skeleton_labels=False):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = [
        'nose',
        'l_eye',
        'r_eye',
        'l_ear',
        'r_ear',
        'l_shoulder',  # 5
        'r_shoulder',
        'l_elbow',
        'r_elbow',
        'l_wrist',
        'r_wrist',  # 10
        'l_hip',
        'r_hip',
        'l_knee',
        'r_knee',
        'l_ankle',
        'r_ankle'
    ]

    for i, j in skeleton:
        if kp[i - 1][0] >= 0 and kp[i - 1][1] >= 0 and kp[
                j - 1][0] >= 0 and kp[j - 1][1] >= 0 and (
                    len(kp[i - 1]) <= 2 or
                    (len(kp[i - 1]) > 2 and kp[i - 1][2] > 0.1
                     and kp[j - 1][2] > 0.1)):
            cv2.line(aa, tuple(kp[i - 1][:2]), tuple(kp[j - 1][:2]),
                     (0, 255, 255), 2)
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                cv2.circle(aa, tuple(kp[j][:2]), 1, tuple((0, 0, 255)), -1)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                cv2.circle(aa, tuple(kp[j][:2]), 1, tuple((255, 0, 0)), -1)

            if show_skeleton_labels and (len(kp[j]) <= 2 or
                                         (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))


def visualize(img: np.ndarray,
              det_box_list=None,
              gt_box_list=None,
              kps_list=None,
              show_skeleton_labels=False) -> np.ndarray:
    """Draw bbox and keypoints on image.

    Args:
        img (np.ndarray): BGR
        det_box_list (list, optional): Defaults to None.
        gt_box_list (list, optional): Defaults to None.
        kps_list (list, optional): Defaults to None.
        show_skeleton_labels (bool, optional): Defaults to False.

    Returns:
        np.ndarray: BGR
    """
    im = np.array(img).copy().astype(np.uint8)
    if det_box_list:
        for det_boxes in det_box_list:
            det_boxes = np.array(det_boxes)
            bb = det_boxes[:4].astype(int)
            cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

    if gt_box_list:
        for gt_boxes in gt_box_list:
            gt_boxes = np.array(gt_boxes)
            for gt in gt_boxes:
                bb = gt[:4].astype(int)
                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255),
                              2)
    if kps_list:
        for kps in kps_list:
            kps = np.array(kps).astype(int)
            try:
                kps = kps.reshape(-1, 17, 3)
            except ValueError:
                kps = kps.reshape(-1, 17, 2)

            for i in range(len(kps)):
                draw_skeleton(im, kps[i], show_skeleton_labels)
    return im.copy()


def show_panel_mem(dataset,
                   frame_id,
                   multi_kps3d,
                   data_name,
                   img_folder,
                   output_dir='./result'):
    if multi_kps3d.shape[2] == 3:
        multi_kps3d = multi_kps3d.transpose(0, 2, 1)
    try:
        show_panel_mem.counter += 1
    except AttributeError:
        show_panel_mem.counter = 0

    Ps = dataset.P
    reprojected_kps = list()
    for camId, P in enumerate(Ps):
        reprojected_kps.append([])
        for kps3d in multi_kps3d:
            kps3dHomo = np.ones((4, kps3d.shape[1]))
            kps3dHomo[:3] = kps3d
            kps2dHomo = P @ kps3dHomo
            kps2dHomo /= kps2dHomo[2]
            reprojected_kps[camId].append(kps2dHomo.T)

    for i, cam_id in enumerate(dataset.cam_names):

        if data_name == 'shelf':
            img = cv2.imread(f'{img_folder}/shelf/img/' +
                             f'Camera{cam_id}/img_{frame_id:06d}.png')
        elif data_name == 'campus':
            img = cv2.imread(f'{img_folder}/campus/img/Camera{cam_id}/' +
                             f'campus4-c{cam_id}-{frame_id:05d}.png')
        elif 'panoptic' in data_name:
            img = cv2.imread(f'{img_folder}/{data_name}/img/Camera{cam_id}/' +
                             f'frame_{frame_id:06d}.png')
        else:
            NotImplementedError
        info_dict = copy.deepcopy(dataset.info_dict[cam_id][frame_id])
        img_detected = visualize(
            img,
            kps_list=[i['pose2d'] for i in info_dict],
            det_box_list=[i['bbox'] for i in info_dict])
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f'{output_dir}/cam{cam_id}_frame{frame_id}_detect.png',
                    img_detected)

        img_projected = visualize(img, kps_list=reprojected_kps[i])
        cv2.imwrite(f'{output_dir}/cam{cam_id}_frame{frame_id}_project.png',
                    img_projected)


def draw_skeleton_paper(img, kp, colors):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for idx, (i, j) in enumerate(skeleton):
        if kp[i - 1][0] >= 0 and kp[i - 1][1] >= 0 and kp[
                j - 1][0] >= 0 and kp[j - 1][1] >= 0 and (
                    len(kp[i - 1]) <= 2 or
                    (len(kp[i - 1]) > 2 and kp[i - 1][2] > 0.1
                     and kp[j - 1][2] > 0.1)):
            cv2.line(img, tuple(kp[i - 1][:2]), tuple(kp[j - 1][:2]), colors,
                     5)
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                cv2.circle(img, tuple(kp[j][:2]), 5, tuple(colors), -1)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                cv2.circle(img, tuple(kp[j][:2]), 5, tuple(colors), -1)


def visualize_skeleton_paper(
    img,
    colors,
    det_box_list=None,
    kps_list=None,
):
    im = np.array(img).copy().astype(np.uint8)
    if det_box_list:
        for boxIdx, det_boxes in enumerate(det_box_list):
            det_boxes = np.array(det_boxes)
            bb = det_boxes[:4].astype(int)
            cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), colors[boxIdx],
                          5)

    if kps_list:
        for pid, kps in enumerate(kps_list):
            kps = np.array(kps).astype(int)
            kps = kps.reshape(1, kps.shape[0], -1)

            for i in range(len(kps)):
                draw_skeleton_paper(im, kps[i], colors[pid])

    return im.copy()


def plot_paper3d(kps3d, name, colors):
    """Plot the 3D keypoints showing the keypoints connections.

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    """
    if 'panoptic' in name:
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    else:
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    kps3d = [R @ i for i in kps3d]
    _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                   [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                   [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    fig = plt.figure()

    ax = fig.gca(projection='3d')

    smallest = [min([i[idx].min() for i in kps3d]) for idx in range(3)]
    largest = [max([i[idx].max() for i in kps3d]) for idx in range(3)]
    ax.set_xlim3d(smallest[0], largest[0])
    ax.set_ylim3d(smallest[1], largest[1])
    ax.set_zlim3d(smallest[2], largest[2])

    for i, kp3d in enumerate(kps3d):
        assert (kp3d.ndim == 2)
        assert (kp3d.shape[0] == 3)
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.plot([kp3d[0, c[0]], kp3d[0, c[1]]],
                    [kp3d[1, c[0]], kp3d[1, c[1]]],
                    [kp3d[2, c[0]], kp3d[2, c[1]]],
                    c=col)
        for j in range(kp3d.shape[1]):
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.scatter(
                kp3d[0, j],
                kp3d[1, j],
                kp3d[2, j],
                c=col,
                marker='o',
                edgecolor=col)
    return fig


def plot_paper_rows(dataset,
                    matched_list,
                    sub_imgid2cam,
                    frame_id,
                    multi_kps3d,
                    data_name,
                    img_folder,
                    saveImg=False,
                    output_dir='./result'):
    if multi_kps3d.shape[2] == 3:
        multi_kps3d = multi_kps3d.transpose(0, 2, 1)
    try:
        show_panel_mem.counter += 1
    except AttributeError:
        show_panel_mem.counter = 0
    Ps = dataset.P
    reprojected_kps = list()
    for camId, P in enumerate(Ps):
        reprojected_kps.append([])
        for kp3d in multi_kps3d:
            kp3dHomo = np.ones((4, kp3d.shape[1]))
            kp3dHomo[:3] = kp3d
            kp2dHomo = P @ kp3dHomo
            kp2dHomo /= kp2dHomo[2]
            reprojected_kps[camId].append(kp2dHomo.T)

    def subImgID2Pid(pid):
        for gPid, subImgIds in enumerate(matched_list):
            if pid in subImgIds:
                return gPid

    for cam_idx, cam_id in enumerate(dataset.cam_names):
        info_dict = copy.deepcopy(dataset.info_dict[cam_id][frame_id])
        kps_id_in_camera = [
            subImgID2Pid(idx) for idx, cam_id in enumerate(sub_imgid2cam)
            if cam_id == cam_id
        ]
        color_assignment = [_ALL_COLOR[pid_g] for pid_g in kps_id_in_camera]
        if data_name == 'shelf':
            img = cv2.imread(f'{img_folder}/shelf/img/' +
                             f'Camera{cam_id}/img_{frame_id:06d}.png')
        elif data_name == 'campus':
            img = cv2.imread(f'{img_folder}/campus/img/Camera{cam_id}/' +
                             f'campus4-c{cam_id}-{frame_id:05d}.png')
        elif 'panoptic' in data_name:
            img = cv2.imread(f'{img_folder}/{data_name}/img/Camera{cam_id}/' +
                             f'frame_{frame_id:06d}.png')
        else:
            NotImplementedError
        img_detected = visualize_skeleton_paper(
            img, [(255, 255, 255) for _ in info_dict],
            kps_list=[i['pose2d'] for i in info_dict],
            det_box_list=[i['bbox'] for i in info_dict])
        img_matched = visualize_skeleton_paper(
            img,
            color_assignment,
            kps_list=[i['pose2d'] for i in info_dict],
            det_box_list=[i['bbox'] for i in info_dict])
        img_projected = visualize_skeleton_paper(
            img, _ALL_COLOR, kps_list=reprojected_kps[cam_idx])
        if saveImg:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(
                f'{output_dir}/cam{cam_idx}_frame{frame_id}_detected.png',
                img_detected)
            cv2.imwrite(
                f'{output_dir}/cam{cam_idx}_frame{frame_id}_matched.png',
                img_matched)
            cv2.imwrite(
                f'{output_dir}/cam{cam_idx}_frame{frame_id}_projected.png',
                img_projected)

    if multi_kps3d is not None:
        fig = plot_paper3d(multi_kps3d, data_name, _ALL_COLOR)
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/panel_{frame_id}.png')
        plt.close(fig)


def plot_kpt(keypoint, canvas, is_homo=False, match_id=None):
    for i, (x, y) in enumerate(keypoint[:, 0:2]):
        if np.isnan(x) or np.isnan(y):
            continue

        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0] and is_homo:
            # matched
            if match_id is not None:
                cv2.circle(
                    canvas, (int(x), int(y)),
                    14,
                    _ALL_COLOR[match_id[i][1] % len(_ALL_COLOR)],
                    thickness=-1)
                cv2.circle(
                    canvas, (int(x), int(y)), 5, _ALL_COLOR[100], thickness=-1)
                continue
            # homo
            cv2.circle(
                canvas, (int(x), int(y)),
                14,
                _ALL_COLOR[i % len(_ALL_COLOR)],
                thickness=-1)
            continue

        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            cv2.circle(
                canvas, (int(x), int(y)),
                7,
                _ALL_COLOR[i % len(_ALL_COLOR)],
                thickness=-1)
    return canvas


def plot_and_encode(keypoints2d,
                    frame_dir,
                    encode_path,
                    fps=60,
                    start_frame=0,
                    end_frame=-1):
    assert len(keypoints2d.shape
               ) == 3, f'Input shape is not valid! Got {keypoints2d.shape}'
    all_frame_name_list = os.listdir(frame_dir)
    all_frame_name_list.sort()

    frame_name_list = []
    for i, file_name in enumerate(all_frame_name_list):
        if str(start_frame) in file_name:
            frame_name_list = all_frame_name_list[i:i + end_frame -
                                                  start_frame]
            break
    frame_mat = cv2.imread(os.path.join(frame_dir, frame_name_list[0]))
    H, W = frame_mat.shape[0], frame_mat.shape[1]
    video = np.ones((end_frame - start_frame, H, W, 3), dtype=np.uint8) * 255
    for frame_index, frame_name in enumerate(frame_name_list):
        frame_mat = cv2.imread(os.path.join(frame_dir, frame_name))
        video[frame_index, :, :, :] = frame_mat

    if end_frame > start_frame:
        start_frame = 0
        end_frame = min(end_frame, video.shape[0])
    else:
        end_frame = min(video.shape[0], keypoints2d.shape[0])
    keypoint2d_slice = keypoints2d
    video_slice = video[start_frame:end_frame, :, :, :]

    for frame_index, keypoint in enumerate(keypoint2d_slice):
        if frame_index >= video_slice.shape[0]:
            break
        video_slice[frame_index] = plot_kpt(keypoint, video_slice[frame_index])

    array_to_video(video_slice, encode_path, fps=fps, disable_log=True)
