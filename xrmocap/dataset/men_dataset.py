import cv2
import json
import numpy as np
import os.path as osp
import sys
import torch
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx
from xrprimer.data_structure.camera.fisheye_camera import \
    FisheyeCameraParameter  # FisheyeCamera with distortion

# Config project if not exist
project_path = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)


class MemDataset(Dataset):
    """Datasets in memory to boost performance of whole pipeline."""

    def __init__(self,
                 info_dict,
                 cam_param_list=None,
                 template_name='Shelf',
                 homo_folder=None):
        self.args = dict(height=256, width=128)
        self.normalizer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.test_transformer = T.Compose([
            T.Resize((self.args['height'], self.args['width']),
                     interpolation=3),
            T.ToTensor(),
            self.normalizer,
        ])
        self.info_dict = info_dict
        self.cam_names = sorted(info_dict.keys())

        self.dimGroup = OrderedDict()

        for img_id in [
                i for i in self.info_dict[self.cam_names[0]].keys()
                if i != 'image_data' and i != 'image_path'
        ]:
            cnt = 0
            this_dim = [0]
            for cam_id in self.cam_names:
                n_person = len(self.info_dict[cam_id][img_id])
                cnt += n_person
                this_dim.append(cnt)
            self.dimGroup[int(img_id)] = torch.Tensor(this_dim).long()

        # handle camera parameter
        n_cameras = len(cam_param_list)
        camera_parameter = {
            'P': np.zeros((n_cameras, 3, 4)),
            'K': np.zeros((n_cameras, 3, 3)),
            'RT': np.zeros((n_cameras, 3, 4)),
            'H': np.zeros((n_cameras, 3, 3))
        }
        distCoeff = np.zeros((n_cameras, 5))
        for i, input_cam_param in enumerate(cam_param_list):
            if issubclass(input_cam_param.__class__, FisheyeCameraParameter):
                cam_param = input_cam_param.clone()
            else:
                raise TypeError
            camera_parameter['K'][i] = cam_param.get_intrinsic(k_dim=3)
            camera_parameter['RT'][i] = np.concatenate(
                (np.asarray(cam_param.extrinsic_r),
                 np.asarray(cam_param.extrinsic_t)[:, np.newaxis]),
                axis=1)
            # compute projection matrix
            proj_mat = camera_parameter['K'][i] @ camera_parameter['RT'][i]
            camera_parameter['P'][i] = proj_mat
            if homo_folder is not None:
                try:
                    homo = json.load(
                        open(osp.join(homo_folder,
                                      f'cam{i}.json')))['homography']['data']
                except FileNotFoundError:
                    raise FileNotFoundError
            else:
                homo = np.zeros((3, 3))
            camera_parameter['H'][i] = np.array(homo).reshape(3, 3)

            distCoeff[i] = [
                cam_param.k1, cam_param.k2, cam_param.p1, cam_param.p2,
                cam_param.k3
            ]
        self.P = camera_parameter['P'].astype(np.float32)
        self.K = camera_parameter['K'].astype(np.float32)
        self.RT = camera_parameter['RT'].astype(np.float32)
        self.H = camera_parameter['H'].astype(np.float32)
        self.distCoeff = distCoeff.astype(np.float32)

        if homo_folder is not None:
            # calculate the projection for active frames
            for camera_id in self.info_dict.keys():
                for img_id in [
                        i for i in self.info_dict[self.cam_names[0]].keys()
                        if i != 'image_data' and i != 'image_path'
                ]:
                    if img_id not in self.info_dict[camera_id]:
                        continue
                    if len(self.info_dict[camera_id][img_id]) == 0:
                        continue
                    # use left feet toe to calculate H matrix,
                    index = get_keypoint_idx(
                        name='left_ankle', convention='coco')
                    feet = np.asarray([
                        info['pose2d'][index]
                        for info in self.info_dict[camera_id][img_id]
                    ])[:, :2]
                    points = feet.astype(np.float32).reshape(-1, 1, 2)
                    proj = cv2.perspectiveTransform(points, self.H[camera_id])
                    proj = np.where(proj == 0, np.nan, proj)
                    for human_idx in range(
                            len(self.info_dict[camera_id][img_id])):
                        self.info_dict[camera_id][img_id][human_idx][
                            'h_proj'] = proj[human_idx]

        # calculate the fundamental matrix for geometry affinity
        self.skew_op = lambda x: torch.tensor([[0, -x[2], x[
            1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

        self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1:\
            torch.inverse(K_0).t() @ (R_0 @ R_1.t()) @ K_1.t() @ \
            self.skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))

        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1:\
            self.fundamental_op(
                K_0, RT_0[:, :3], RT_0[:, 3], K_1, RT_1[:, :3], RT_1[:, 3])

        self.F = torch.zeros(len(self.cam_names), len(self.cam_names), 3,
                             3)  # NxNx3x3 matrix
        # TODO: optimize this stupid nested for loop
        for i in range(len(self.cam_names)):
            for j in range(len(self.cam_names)):
                self.F[i, j] += self.fundamental_RT_op(
                    torch.tensor(self.K[i]), torch.tensor(self.RT[i]),
                    torch.tensor(self.K[j]), torch.tensor(self.RT[j]))
                if self.F[i, j].sum() == 0:
                    self.F[i, j] += 1e-12  # to avoid nan
        # handle heatmap info
        self.heatmaps = None
        self.template = load_template()
        self.distribution = load_distribution(template_name)

    def __getitem__(self, item):
        """Get a list of image in multi view at the same time.

        Args:
            item (int): frame id

        Returns:
            tuple (images, fnames, pid, cam_id)
        """
        img_id = item
        data_by_cam = OrderedDict()
        for cam_id in self.cam_names:
            data_by_cam[cam_id] = [
                v['cropped_img'] for v in self.info_dict[cam_id][img_id]
            ]
        image = list()
        fname = list()
        pid = list()
        cam_id = list()
        for k, v in data_by_cam.items():
            for i, _ in enumerate(v):
                fname += [f'{k}_{i}']
            pid += list(range(len(v)))
            cam_id += [k for i in v]
            image += [
                self.test_transformer(Image.fromarray(np.uint8(i))) for i in v
            ]
        image = torch.stack(image)
        data_batch = (image, fname, pid, cam_id)
        return data_batch

    def __len__(self):
        if len(self.info_dict):
            return len(self.info_dict[self.cam_names[0]])
        else:
            return 0

    def get_tracking_data(self, img_id, n_kps2d):
        """Get data in multi view at the same time (current frame)

        :param img_id:
        :return: H projected points, keypoints, trackid, camera group, img_id
        """
        max_proposal = 0
        for cam_id in self.cam_names:
            proposal = len(self.info_dict[cam_id][img_id])
            if proposal > max_proposal:
                max_proposal = proposal

        homo_points = np.full((len(self.cam_names), max_proposal + 1, 2),
                              np.nan)
        mview_kps2d = np.full(
            (len(self.cam_names), max_proposal + 1, n_kps2d, 2), np.nan)
        track_id = np.full((len(self.cam_names), max_proposal + 1), np.nan)

        for cam_id in self.cam_names:
            for person_id in range(len(self.info_dict[cam_id][img_id])):
                homo_points[cam_id, person_id] = self.info_dict[cam_id][
                    img_id][person_id]['h_proj']
                mview_kps2d[cam_id, person_id] = self.info_dict[cam_id][
                    img_id][person_id]['pose2d'][:, :2]
                track_id[cam_id, person_id] = self.info_dict[cam_id][img_id][
                    person_id]['id']

        return homo_points, mview_kps2d, track_id, img_id

    def get_unary(self, person, sub_imgid2cam, candidates, img_id):

        def get2Dfrom3D(x, P):
            """Get the 2d keypoints from 3d keypoints."""
            x4d = np.append(x, 1)
            x2d = np.dot(P, x4d)[0:2] / (np.dot(P, x4d)[2] + 10e-6
                                         )  # to avoid np.dot(P, x4d)[2] = 0

            return x2d

        # get the unary of 3D candidates
        n_kps2d = len(candidates)
        n_point = len(candidates[0])
        unary = np.ones((n_kps2d, n_point))
        info_list = list()  # This also occur in multi estimater
        for cam_id in self.cam_names:
            info_list += self.info_dict[cam_id][img_id]
        # project the 3d point to each view to get the 2d points

        for pid in person:
            use_heatmap = 'heatmap_data' in info_list[pid]
            Pi = self.P[sub_imgid2cam[pid]]
            if n_kps2d == 19 or not use_heatmap:  # omni
                kps_2d = info_list[pid]['pose2d']
                kps_conf = info_list[pid]['conf']
            elif n_kps2d == 17 and use_heatmap:  # coco
                heatmap = info_list[pid]['heatmap_data']
                crop = np.array(info_list[pid]['heatmap_bbox'])
            else:
                raise NotImplementedError
            points_3d = candidates.reshape(-1, 3).T
            points_3d_homo = np.vstack(
                [points_3d,
                 np.ones(points_3d.shape[-1]).reshape(1, -1)])
            points_2d_homo = (Pi @ points_3d_homo).T.reshape(n_kps2d, -1, 3)
            points_2d = points_2d_homo[..., :2] / (
                points_2d_homo[..., 2].reshape(n_kps2d, -1, 1) + 10e-6)

            if n_kps2d == 19 or not use_heatmap:
                for kps2d_id, kpj_2d in enumerate(kps_2d):
                    for j, point3d in enumerate(candidates[kps2d_id]):
                        point_2d = points_2d[kps2d_id, j]
                        # we use gaussian to approx the heatmap
                        if np.isnan(kpj_2d).any():
                            unary_i = 10e-6
                        else:
                            pixel_distance = ((kpj_2d - point_2d)**2).sum()
                            unary_i = np.exp(
                                -pixel_distance / 625) * kps_conf[kps2d_id]
                            unary_i = np.clip(unary_i, 10e-6, 1)

                        if np.isnan(unary_i):  # nan in kpj_2d!
                            import pdb
                            pdb.set_trace()
                        unary[kps2d_id, j] = unary[kps2d_id, j] * unary_i
            elif n_kps2d == 17 and use_heatmap:
                for kps2d_id, heatmap_j in enumerate(heatmap):
                    for j, point3d in enumerate(candidates[kps2d_id]):
                        point_2d = points_2d[kps2d_id, j]
                        point_2d_in_heatmap = point_2d - np.array(
                            [crop[0], crop[1]])

                        if point_2d_in_heatmap[0] > heatmap_j.shape[
                                1] or point_2d_in_heatmap[
                                    0] < 0 or point_2d_in_heatmap[
                                        1] > heatmap_j.shape[
                                            0] or point_2d_in_heatmap[1] < 0:
                            unary_i = 10e-6
                        else:
                            unary_i = heatmap_j[int(point_2d_in_heatmap[1]),
                                                int(point_2d_in_heatmap[0])]
                        unary[kps2d_id, j] = unary[kps2d_id, j] * unary_i
            else:
                raise NotImplementedError
        unary = np.log10(unary)
        return unary


def load_template(dataset='h36m'):
    """
    Hard encode the human body template
    :return:
    """
    templates = {
        'h36m':
        np.array([[
            0.0018327, 0.18507086, -0.17760321, 0.47678296, -0.46611124,
            0.71017444, -0.71153766, 0.11616346, -0.12763677, 0.11020779,
            -0.12279839, 0.12724847, -0.12452087
        ],
                  [
                      -0.0827738, -0.07526917, -0.05761691, -0.09604145,
                      -0.02306564, -0.18181808, -0.06122154, -0.12290852,
                      -0.09051553, -0.08240831, -0.0523845, 0.03715071,
                      0.05312368
                  ],
                  [
                      1.70503833, 1.48879248, 1.4854071, 1.44106006,
                      1.42731128, 1.42766638, 1.40946619, 0.97231879,
                      1.00533917, 0.50190244, 0.53471307, 0.04910713,
                      0.07812376
                  ]]),
        'Shelf':
        np.array([[
            0.01273053, -0.09262084, -0.11961558, -0.07061234, -0.08761829,
            0.05067334, 0.0088842, 0.02459383, -0.08589214, 0.05839888,
            -0.08001912, -0.00395661, -0.14304384
        ],
                  [
                      0.05546921, 0.22573541, -0.11484059, 0.25385895,
                      -0.20887429, 0.1862903, -0.16983723, 0.15736914,
                      -0.06168539, 0.16666036, -0.06817156, 0.1914962,
                      -0.09228449
                  ],
                  [
                      1.60827349, 1.28002543, 1.28858008, 1.00131741,
                      1.00584484, 0.82851737, 0.7909359, 0.75035656,
                      0.73453197, 0.3672495, 0.38460963, -0.04995751,
                      -0.04118636
                  ]]),
        'Campus':
        np.array([[
            -0.52248502, -0.64536842, -0.37618539, -0.64643804, -0.28080107,
            -0.61725263, -0.39121596, -0.53340433, -0.42570307, -0.47950823,
            -0.33426481, -0.46441123, -0.45108205
        ],
                  [
                      4.01057597, 3.88068601, 3.85644611, 3.88494234,
                      3.90516631, 4.05613315, 4.02384458, 3.81515482,
                      3.85981597, 3.93538466, 3.81045037, 3.89418933,
                      3.48824897
                  ],
                  [
                      1.95452321, 1.65249654, 1.63991337, 1.32163371,
                      1.27597037, 1.30090807, 1.21906915, 1.04422362,
                      1.02544295, 0.57991175, 0.58941852, 0.07508519,
                      0.30164174
                  ]])
    }
    return templates[dataset]


def load_distribution(dataset='Unified'):
    joints2edges = {
        (0, 1): 0,
        (1, 0): 0,
        (0, 2): 1,
        (2, 0): 1,
        (0, 7): 2,
        (7, 0): 2,
        (0, 8): 3,
        (8, 0): 3,
        (1, 3): 4,
        (3, 1): 4,
        (2, 4): 5,
        (4, 2): 5,
        (3, 5): 6,
        (5, 3): 6,
        (4, 6): 7,
        (6, 4): 7,
        (7, 9): 8,
        (9, 7): 8,
        (8, 10): 9,
        (10, 8): 9,
        (9, 11): 10,
        (11, 9): 10,
        (10, 12): 11,
        (12, 10): 11
    }
    distribution_dict = {
        'Shelf': {
            'mean':
            np.array([
                0.30280354, 0.30138756, 0.79123502, 0.79222949, 0.28964179,
                0.30393598, 0.24479075, 0.24903801, 0.40435882, 0.39445121,
                0.3843522, 0.38199836
            ]),
            'std':
            np.array([
                0.0376412, 0.0304385, 0.0368604, 0.0350577, 0.03475468,
                0.03876828, 0.0353617, 0.04009757, 0.03974647, 0.03696424,
                0.03008979, 0.03143456
            ]) * 2,
            'joints2edges':
            joints2edges
        },
        'Campus': {
            'mean':
            np.array([
                0.29567343, 0.28090078, 0.89299809, 0.88799211, 0.32651703,
                0.33454941, 0.29043165, 0.29932416, 0.43846395, 0.44881553,
                0.46952846, 0.45528477
            ]),
            'std':
            np.array([
                0.01731019, 0.0226062, 0.06650426, 0.06009805, 0.04606478,
                0.04059899, 0.05868499, 0.06553948, 0.04129285, 0.04205624,
                0.03633746, 0.02889456
            ]) * 2,
            'joints2edges':
            joints2edges
        },
        'Unified': {
            'mean':
            np.array([
                0.29743698, 0.28764493, 0.86562234, 0.86257052, 0.31774172,
                0.32603399, 0.27688682, 0.28548218, 0.42981244, 0.43392589,
                0.44601327, 0.43572195
            ]),
            'std':
            np.array([
                0.02486281, 0.02611557, 0.07588978, 0.07094158, 0.04725651,
                0.04132808, 0.05556177, 0.06311393, 0.04445206, 0.04843436,
                0.0510811, 0.04460523
            ]) * 16,
            'joints2edges':
            joints2edges
        }
    }
    # logger.debug(f"Using distribution on {dataset}")
    return distribution_dict[dataset]
