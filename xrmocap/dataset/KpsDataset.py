# yapf: disable
import copy
import cv2
import logging
import numpy as np
import os
from torch.utils.data import Dataset
from xrprimer.data_structure.camera import \
    FisheyeCameraParameter  # Camera with distortion

from xrmocap.transform.point import affine_transform_pts
from xrmocap.utils.geometry import get_affine_transform, get_scale

logger = logging.getLogger(__name__)

# yapf: enable


class KpsDataset(Dataset):
    """Modified from MVP.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 image_set,
                 is_train,
                 image_size,
                 n_max_person,
                 dataset_root,
                 root_idx,
                 dataset,
                 data_format,
                 data_augmentation,
                 sigma,
                 n_cameras,
                 scale_factor,
                 rot_factor,
                 flip,
                 color_rgb,
                 target_type,
                 heatmap_size,
                 use_different_kps_weight,
                 space_size,
                 space_center,
                 initial_cube_size,
                 transform=None):

        self.n_kps = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.maximum_person = n_max_person

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', dataset_root)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = root_idx
        self.image_set = image_set
        self.dataset_name = dataset

        self.data_format = data_format
        self.data_augmentation = data_augmentation

        self.n_views = n_cameras

        self.scale_factor = scale_factor
        self.rotation_factor = rot_factor
        self.flip = flip
        self.color_rgb = color_rgb

        self.target_type = target_type
        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.sigma = sigma
        self.use_different_kps_weight = use_different_kps_weight
        self.kps_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(space_size)
        self.space_center = np.array(space_center)
        self.initial_cube_size = np.array(initial_cube_size)

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if 'source' in db_rec and db_rec['source'] == 'h36m':
            # crop image from 1002 x 1000 to 1000 x 1000 for h36m
            data_numpy = data_numpy[:1000]

        if data_numpy is None:
            return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        kps2d = db_rec['kps2d']
        kps3d = db_rec['kps3d']
        kps2d_vis = db_rec['kps2d_vis']
        kps3d_vis = db_rec['kps3d_vis']

        n_person = len(kps2d)
        assert n_person <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0  # NOTE: do not apply rotation augmentation

        trans = get_affine_transform(c, s, r, self.image_size, inv=0)
        # NOTE: this trans represents full image to cropped image,
        # not full image->heatmap
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(n_person):
            for i in range(len(kps2d[0])):
                if kps2d_vis[n][i, 0] > 0.0:
                    kps2d[n][i, 0:2] = affine_transform_pts(
                        kps2d[n][i, 0:2], trans)
                    if (np.min(kps2d[n][i, :2]) < 0
                            or kps2d[n][i, 0] >= self.image_size[0]
                            or kps2d[n][i, 1] >= self.image_size[1]):
                        kps2d_vis[n][i, :] = 0

        # NOTE: deal with affine transform
        # affine transform between origin img and heatmap
        aff_trans = np.eye(3, 3)
        aff_trans[0:2] = trans  # full img -> cropped img
        inv_aff_trans = np.eye(3, 3)
        inv_trans = get_affine_transform(c, s, r, self.image_size, inv=1)
        inv_aff_trans[0:2] = inv_trans

        # 3x3 data augmentation affine trans (scale rotate=0)
        # NOTE: this transformation contains both heatmap->image scale affine
        # and data augmentation affine
        aug_trans = np.eye(3, 3)
        aug_trans[0:2] = trans  # full img -> cropped img
        hm_scale = self.heatmap_size / self.image_size
        scale_trans = np.eye(3, 3)  # cropped img -> heatmap
        scale_trans[0, 0] = hm_scale[1]
        scale_trans[1, 1] = hm_scale[0]
        aug_trans = scale_trans @ aug_trans
        # NOTE: aug_trans is superset of affine_trans

        # make kps2d and kps2d_vis having same shape
        kps2d_u = np.zeros((self.maximum_person, self.n_kps, 2))
        kps2d_vis_u = np.zeros((self.maximum_person, self.n_kps, 2))
        for i in range(n_person):
            kps2d_u[i] = kps2d[i]
            kps2d_vis_u[i] = kps2d_vis[i]

        kps3d_u = np.zeros((self.maximum_person, self.n_kps, 3))
        kps3d_vis_u = np.zeros((self.maximum_person, self.n_kps, 3))
        for i in range(n_person):
            kps3d_u[i] = kps3d[i][:, 0:3]
            kps3d_vis_u[i] = kps3d_vis[i][:, 0:3]

        if isinstance(self.root_id, int):
            roots_kps3d = kps3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_kps3d = np.mean([kps3d_u[:, j] for j in self.root_id],
                                  axis=0)

        # NOTE: deal with camera
        cam = db_rec['camera']
        cam_intri = np.eye(3, 3)
        cam_intri[0, 0] = float(cam['fx'])
        cam_intri[1, 1] = float(cam['fy'])
        cam_intri[0, 2] = float(cam['cx'])
        cam_intri[1, 2] = float(cam['cy'])
        cam_R = cam['R']
        cam_T = cam['T']
        cam_standard_T = cam['standard_T']

        meta = {
            'image':
            image_file,
            'n_person':
            n_person,
            'kps3d':
            kps3d_u,
            'kps3d_vis':
            kps3d_vis_u,
            'roots_kps3d':
            roots_kps3d,
            'kps2d':
            kps2d_u,
            'kps2d_vis':
            kps2d_vis_u,
            'center':
            c,
            'scale':
            s,
            'rotation':
            r,
            'camera_focal':
            np.stack([cam['fx'], cam['fy'],
                      np.ones_like(cam['fy'])]),
            'camera_T':
            cam_T,
            'affine_trans':
            aff_trans,
            'inv_affine_trans':
            inv_aff_trans,
            'aug_trans':
            aug_trans,
            # To be removed
            'camera':
            cam,
            'camera_Intri':
            cam_intri,
            'camera_R':
            cam_R,
            # for ray direction generation
            'camera_standard_T':
            cam_standard_T,
        }

        # split FisheyeCameraParameter from meta
        dist_coeff_k = []
        dist_coeff_p = []
        cam_param = FisheyeCameraParameter()
        dist_coeff_k = [
            db_rec['camera']['k'][0], db_rec['camera']['k'][1],
            db_rec['camera']['k'][2], 0, 0, 0
        ]
        dist_coeff_p = [db_rec['camera']['p'][0], db_rec['camera']['p'][1]]
        cam_param.set_dist_coeff(dist_coeff_k, dist_coeff_p)

        cam_param.set_KRT(
            K=cam_intri, R=cam['R'], T=cam['standard_T'], world2cam=False)
        cam_param.set_resolution(height=height, width=width)

        # Temp
        end_of_clip = False

        return input, cam_param, end_of_clip, meta

    def compute_human_scale(self, kps2d, kps2d_vis):
        idx = kps2d_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(kps2d[idx, 0]), np.max(kps2d[idx, 0])
        miny, maxy = np.min(kps2d[idx, 1]), np.max(kps2d[idx, 1])
        return np.clip(
            np.maximum(maxy - miny, maxx - minx)**2, 1.0 / 4 * 96**2,
            4 * 96**2)

    def generate_target_heatmap(self, kps2d, kps2d_vis):
        '''
        :param kps2d:  [[n_kps, 3]]
        :param kps2d_vis: [n_kps, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        n_person = len(kps2d)
        n_kps = self.n_kps
        target_weight = np.zeros((n_kps, 1), dtype=np.float32)
        for i in range(n_kps):
            for n in range(n_person):
                if kps2d_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (n_kps, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(n_person):
                human_scale = 2 * self.compute_human_scale(
                    kps2d[n] / feat_stride, kps2d_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for kps_idx in range(n_kps):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(kps2d[n][kps_idx][0] / feat_stride[0])
                    mu_y = int(kps2d[n][kps_idx][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if kps2d_vis[n][kps_idx, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                               (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[kps_idx][img_y[0]:img_y[1], img_x[0]:img_x[1]] \
                        = np.maximum(target[kps_idx][img_y[0]:img_y[1],
                                     img_x[0]:img_x[1]],
                                     g[g_y[0]:g_y[1],
                                     g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_kps_weight:
            target_weight = np.multiply(target_weight, self.kps_weight)

        return target, target_weight

    def generate_3d_target(self, kps3d):
        n_people = len(kps3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2,
                              cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2,
                              cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2,
                              cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]),
                          dtype=np.float32)
        cur_sigma = 200.0

        for n in range(n_people):
            kps_idx = self.root_id  # mid-hip
            if isinstance(kps_idx, int):
                mu_x = kps3d[n][kps_idx][0]
                mu_y = kps3d[n][kps_idx][1]
                mu_z = kps3d[n][kps_idx][2]
            elif isinstance(kps_idx, list):
                mu_x = (kps3d[n][kps_idx[0]][0] +
                        kps3d[n][kps_idx[1]][0]) / 2.0
                mu_y = (kps3d[n][kps_idx[0]][1] +
                        kps3d[n][kps_idx[1]][1]) / 2.0
                mu_z = (kps3d[n][kps_idx[0]][2] +
                        kps3d[n][kps_idx[1]][2]) / 2.0
            i_x = [
                np.searchsorted(grid1Dx, mu_x - 3 * cur_sigma),
                np.searchsorted(grid1Dx, mu_x + 3 * cur_sigma, 'right')
            ]
            i_y = [
                np.searchsorted(grid1Dy, mu_y - 3 * cur_sigma),
                np.searchsorted(grid1Dy, mu_y + 3 * cur_sigma, 'right')
            ]
            i_z = [
                np.searchsorted(grid1Dz, mu_z - 3 * cur_sigma),
                np.searchsorted(grid1Dz, mu_z + 3 * cur_sigma, 'right')
            ]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(
                grid1Dx[i_x[0]:i_x[1]],
                grid1Dy[i_y[0]:i_y[1]],
                grid1Dz[i_z[0]:i_z[1]],
                indexing='ij')
            g = np.exp(-((gridx - mu_x)**2 + (gridy - mu_y)**2 +
                         (gridz - mu_z)**2) / (2 * cur_sigma**2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(
                target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target
