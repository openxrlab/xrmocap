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
from xrprimer.data_structure.camera.fisheye_camera import \
    FisheyeCameraParameter  # FisheyeCamera with distortion

from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx

# Config project if not exist
project_path = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)


class MemDataset(Dataset):
    """Datasets in memory to boost performance of whole pipeline."""

    def __init__(self, info_dict, cam_param_list=None, homo_folder=None):
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
                        info['kps2d'][index]
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
                    img_id][person_id]['kps2d'][:, :2]
                track_id[cam_id, person_id] = self.info_dict[cam_id][img_id][
                    person_id]['id']

        return homo_points, mview_kps2d, track_id, img_id


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
