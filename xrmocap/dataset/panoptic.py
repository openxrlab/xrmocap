import copy
import glob
import json_tricks as json
import logging
import numpy as np
import os
import os.path as osp
import pickle

from xrmocap.dataset.KpsDataset import KpsDataset
from xrmocap.utils.geometry import project_3dpts

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    # '160906_band3',
]
VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']


class Panoptic(KpsDataset):
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
        self.pixel_std = 200.0
        self.n_kps = n_kps

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3

            self.cam_list = [
                (0, 3), (0, 12), (0, 23), (0, 13), (0, 6)
            ][:self.n_views]  # pretrain with 3,12,23 for better campus ft
            self.n_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = [(0, 3), (0, 12), (0, 23), (0, 13),
                             (0, 6)][:self.n_views]
            self.n_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}.pkl'.\
            format(self.image_set, self.n_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        for seq in self.sequence_list:

            cameras = self._get_cam(seq)
            # GT dir for every dataset
            curr_anno = osp.join(self.dataset_root, seq,
                                 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob(f'{curr_anno:s}/*.json'))

            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0:
                        continue

                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = f'{k[0]:02d}_{k[1]:02d}'
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')

                        all_gt_kps3d = []
                        all_gt_kps3d_vis = []
                        all_gt_kps2d = []
                        all_gt_kps2d_vis = []
                        for body in bodies:
                            gt_kps3d = np.array(body['joints19'])\
                                .reshape((-1, 4))
                            gt_kps3d = gt_kps3d[:self.n_kps]

                            kps3d_vis = gt_kps3d[:, -1] > 0.1

                            if not kps3d_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            gt_kps3d[:, 0:3] = gt_kps3d[:, 0:3].dot(M)

                            all_gt_kps3d.append(gt_kps3d[:, 0:3] * 10.0)
                            all_gt_kps3d_vis.append(
                                np.repeat(
                                    np.reshape(kps3d_vis, (-1, 1)), 3, axis=1))

                            gt_kps2d = np.zeros((gt_kps3d.shape[0], 2))
                            gt_kps2d[:, :2] = project_3dpts(
                                gt_kps3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                            x_check = \
                                np.bitwise_and(gt_kps2d[:, 0] >= 0,
                                               gt_kps2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(gt_kps2d[:, 1] >= 0,
                                               gt_kps2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            kps3d_vis[np.logical_not(check)] = 0

                            all_gt_kps2d.append(gt_kps2d)
                            all_gt_kps2d_vis.append(
                                np.repeat(
                                    np.reshape(kps3d_vis, (-1, 1)), 2, axis=1))

                        if len(all_gt_kps3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(v['R'].T,
                                                   v['t']) * 10.0  # cm to mm
                            our_cam['standard_T'] = v['t'] * 10.0
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]]\
                                .reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]]\
                                .reshape(2, 1)

                            postfix_ = postfix.split('.')[0]
                            db.append({
                                'key':
                                f'{seq}_{prefix}{postfix_}',
                                'image':
                                osp.join(self.dataset_root, image),
                                'kps3d':
                                all_gt_kps3d,
                                'kps3d_vis':
                                all_gt_kps3d_vis,
                                'kps2d':
                                all_gt_kps2d,
                                'kps2d_vis':
                                all_gt_kps2d_vis,
                                'camera':
                                our_cam
                            })
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq,
                            f'calibration_{seq:s}.json')
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, meta = [], []
        for k in range(self.n_views):
            i, c, _, m = super().__getitem__(self.n_views * idx + k)
            input.append(i)
            # TODO: get KRT from Camera and return separately
            # currently KRT is inclueded in meta
            meta.append(m)
        return input, meta

    def __len__(self):
        return self.db_size // self.n_views

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.n_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.n_views * i
            db_rec = copy.deepcopy(self.db[index])
            kps3d = db_rec['kps3d']
            kps3d_vis = db_rec['kps3d_vis']

            if len(kps3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pred_kps3d in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(kps3d, kps3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum(
                                (pred_kps3d[vis, 0:3] - gt[vis])**2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pred_kps3d[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += len(kps3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return \
            aps, \
            recs, \
            self._eval_list_to_mpjpe(eval_list), \
            self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt
