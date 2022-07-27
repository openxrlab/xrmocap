import copy
import json_tricks as json
import numpy as np
import os
import os.path as osp
import pickle
import scipy.io as scio
from collections import OrderedDict

from xrmocap.dataset.KpsDataset import KpsDataset
from xrmocap.utils.camera_utils import project_pose


class Campus(KpsDataset):
    """Modified from MVP.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 is_train,
                 image_set,
                 n_kps,
                 n_max_person,
                 dataset_root,
                 root_idx,
                 dataset,
                 data_format,
                 data_augmentation,
                 n_cameras,
                 scale_factor,
                 rot_factor,
                 flip,
                 color_rgb,
                 target_type,
                 image_size,
                 heatmap_size,
                 use_different_kps_weight,
                 space_size,
                 space_center,
                 initial_cube_size,
                 pesudo_gt,
                 sigma,
                 transform=None):
        self.pixel_std = 200.0
        super().__init__(image_set, is_train, image_size, n_max_person,
                         dataset_root, root_idx, dataset, data_format,
                         data_augmentation, sigma, n_cameras, scale_factor,
                         rot_factor, flip, color_rgb, target_type,
                         heatmap_size, use_different_kps_weight, space_size,
                         space_center, initial_cube_size, transform)
        self.n_kps = n_kps
        self.cam_list = [0, 1, 2]
        self.n_views = len(self.cam_list)

        if self.is_train:
            # augmented training set
            self.frame_range = list(range(0, 350)) + list(
                range(471, 650)) + list(range(751, 1900)) + list(
                    range(471, 520)) * 2 + list(range(751, 1200)) * 2
        else:
            self.frame_range = list(range(350, 471)) \
                               + list(range(650, 751))

        self.db = self._get_db(
            osp.join('./data/CampusSeq1/pesudo_gt/', pesudo_gt))

        self.db_size = len(self.db)

    def _get_db(self, pesudo_gt_path):
        width = 360
        height = 288

        db = []
        cameras = self._get_cam()

        datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        data = scio.loadmat(datafile)
        actors = np.array(np.array(
            data['actor3D'].tolist()).tolist()).squeeze()

        n_person = len(actors)

        if self.is_train:
            with open(pesudo_gt_path, 'rb') as handle:
                gt_voxelpose_infered = pickle.load(handle)
            filtered_frame_range = []
            for i in self.frame_range:
                image = osp.join('Camera0', f'campus4-c0-{i:05d}.png')
                if image.split('/')[-1] in gt_voxelpose_infered:
                    filtered_frame_range.append(i)
            self.frame_range = filtered_frame_range

        for i in self.frame_range:
            for k, cam in cameras.items():
                image = osp.join(f'Camera{k}', f'campus4-c{k}-{i:05d}.png')

                all_gt_kps3d = []
                all_gt_kps3d_vis = []
                all_gt_kps2d = []
                all_gt_kps2d_vis = []
                all_gt_kps3d_conf = []
                if self.is_train:
                    for gt_kps3d in \
                            gt_voxelpose_infered[
                                f'campus4-c{0}-{i:05d}.png']:
                        if len(gt_kps3d[0]) > 0:
                            conf = gt_kps3d[:, 3]
                            gt_kps3d = gt_kps3d[:, :3]
                            all_gt_kps3d_conf.append(conf)
                            all_gt_kps3d.append(gt_kps3d)
                            all_gt_kps3d_vis.append(np.ones((self.n_kps, 3)))

                            gt_kps2d = project_pose(gt_kps3d, cam)

                            x_check = \
                                np.bitwise_and(gt_kps2d[:, 0] >= 0,
                                               gt_kps2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(gt_kps2d[:, 1] >= 0,
                                               gt_kps2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)

                            kps_vis = np.ones((len(gt_kps2d), 1))
                            kps_vis[np.logical_not(check)] = 0
                            all_gt_kps2d.append(gt_kps2d)
                            all_gt_kps2d_vis.append(
                                np.repeat(
                                    np.reshape(kps_vis, (-1, 1)), 2, axis=1))
                else:
                    for person in range(n_person):
                        gt_kps3d = actors[person][i] * 1000.0
                        if len(gt_kps3d[0]) > 0:
                            all_gt_kps3d.append(gt_kps3d)
                            all_gt_kps3d_conf.append(np.array([person] * 14))
                            all_gt_kps3d_vis.append(np.ones((self.n_kps, 3)))

                            gt_kps2d = project_pose(gt_kps3d, cam)

                            x_check = \
                                np.bitwise_and(gt_kps2d[:, 0] >= 0,
                                               gt_kps2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(gt_kps2d[:, 1] >= 0,
                                               gt_kps2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)

                            kps_vis = np.ones((len(gt_kps2d), 1))
                            kps_vis[np.logical_not(check)] = 0
                            all_gt_kps2d.append(gt_kps2d)
                            all_gt_kps2d_vis.append(
                                np.repeat(
                                    np.reshape(kps_vis, (-1, 1)), 2, axis=1))

                # add standard T
                cam['standard_T'] = np.dot(-cam['R'], cam['T'])

                db.append({
                    'image': osp.join(self.dataset_root, image),
                    'kps3d': all_gt_kps3d,
                    'kps3d_vis': all_gt_kps3d_vis,
                    'kps2d': all_gt_kps2d,
                    'kps2d_vis': all_gt_kps2d_vis,
                    'camera': cam,
                })
        return db

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, 'calibration_campus.json')
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras

    def __getitem__(self, idx):
        input, meta = [], []
        for k in range(self.n_views):
            i, c, _, m = super().__getitem__(self.n_views * idx + k)
            input.append(i)
            # cam_list.append(c)
            meta.append(m)
        return input, meta

    def __len__(self):
        return self.db_size // self.n_views

    def evaluate(self, preds, recall_threshold=500):
        datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        data = scio.loadmat(datafile)
        actors = np.array(np.array(
            data['actor3D'].tolist()).tolist()).squeeze()
        n_person = len(actors)

        total_gt = 0
        match_gt = 0

        limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10],
                 [10, 11], [12, 13]]
        correct_parts = np.zeros(n_person)
        total_parts = np.zeros(n_person)
        alpha = 0.5
        limb_correct_parts = np.zeros((n_person, 10))

        for i, fi in enumerate(self.frame_range):
            pred_coco = preds[i].copy()
            pred_coco = pred_coco[pred_coco[:, 0, 3] >= 0, :, :3]
            pred = np.stack([p for p in copy.deepcopy(pred_coco[:, :, :3])])

            for person in range(n_person):
                gt = actors[person][fi] * 1000.0
                if len(gt[0]) == 0:
                    continue

                mpjpes = np.mean(
                    np.sqrt(np.sum((gt[np.newaxis] - pred)**2, axis=-1)),
                    axis=-1)
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0], 0:3] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1], 0:3] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        limb_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2, 0:3] + pred[min_n, 3, 0:3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12, 0:3] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    limb_correct_parts[person, 9] += 1

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        limb_group = OrderedDict([('Head', [8]), ('Torso', [9]),
                                  ('Upper arms', [5,
                                                  6]), ('Lower arms', [4, 7]),
                                  ('Upper legs', [1, 2]),
                                  ('Lower legs', [0, 3])])
        limb_person_pcp = OrderedDict()
        for k, v in limb_group.items():
            limb_person_pcp[k] = \
                np.sum(
                    limb_correct_parts[:, v], axis=-1) \
                / (total_parts / 10 * len(v) + 1e-8)

        return \
            actor_pcp, \
            avg_pcp, \
            limb_person_pcp, \
            match_gt / (total_gt + 1e-8)
