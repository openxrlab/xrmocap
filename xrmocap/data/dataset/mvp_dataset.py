# yapf: disable
import copy
import logging
import numpy as np
import torch
from typing import Tuple, Union

from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx
from xrmocap.transform.point import affine_transform_pts
from xrmocap.utils.geometry import (
    get_affine_transform, get_scale, project_3dpts,
)
from .mview_mperson_dataset import MviewMpersonDataset

# yapf: enable


class MVPDataset(MviewMpersonDataset):

    def __init__(self,
                 dataset: str,
                 data_root: str,
                 img_pipeline: list,
                 image_size: list,
                 heatmap_size: list,
                 kps_thr: float = 0.1,
                 root_kp: Union[None, str] = None,
                 metric_unit: str = 'meter',
                 meta_path: str = 'xrmocap_meta',
                 test_mode: bool = True,
                 shuffled: bool = False,
                 gt_kps3d_convention: Union[None, str] = None,
                 cam_world2cam: bool = False,
                 logger: Union[None, str, logging.Logger] = None,
                 n_max_person: int = 10,
                 n_views: int = 5,
                 n_kps: int = 15) -> None:
        """Init dataset for multi-view pose transformer.

        Args:
            dataset (str):
                Dataset name.
            data_root (str):
                Root path of the downloaded dataset.
            img_pipeline (list):
                A list of image transform instances.
            image_size (list):
                A list of image size.
            heatmap_size (list):
                A list of heatmap size.
            kps_thr (float, optional):
                Threshold for keypoints visibility. Defaults to 0.1.
            root_kp (Union[None, str], optional):
                Root keypoint name. Defaults to None.
            metric_unit (Literal[
                    'meter', 'centimeter', 'millimeter'], optional):
                Metric unit of gt3d and camera parameters. Defaults to 'meter'.
            meta_path (str, optional):
                Path to the train meta-data dir. Defaults to 'xrmocap_meta'.
            test_mode (bool, optional):
                Whether this dataset is used to load testset. Defaults to True.
            shuffled (bool, optional):
                Whether this dataset is used to load shuffled frames.
                If True, getitem will always get end_of_clip=True.
                Defaults to False.
            gt_kps3d_convention (Union[None, str], optional):
                Target convention of keypoints3d, if None,
                kps3d will keep its convention in meta-data.
                Defaults to None.
            cam_world2cam (bool, optional):
                Direction of returned camera extrinsics.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
            n_max_person (int, optional):
                The max number of person for mvp to predict. Defaults to 10.
            n_views (int, optional):
                Number of views. Defaults to 5.
            n_kps (int, optional):
                Number of keypoints. Defaults to 15.
        """
        super().__init__(
            data_root=data_root,
            img_pipeline=img_pipeline,
            meta_path=meta_path,
            test_mode=test_mode,
            shuffled=shuffled,
            metric_unit=metric_unit,
            gt_kps3d_convention=gt_kps3d_convention,
            cam_world2cam=cam_world2cam,
            logger=logger)

        self.test_mode = test_mode
        self.dataset = dataset
        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.maximum_person = n_max_person
        self.root_id = get_keypoint_idx(root_kp, gt_kps3d_convention) \
            if root_kp is not None else None
        self.n_views = n_views
        self.n_kps = n_kps
        self.kps_thr = kps_thr

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, bool, dict]:
        meta = []

        while True:
            mview_img_tensor, k_tensor, r_tensor, t_tensor, kps3d, \
                kw_data, n_person, skip = self.get_one_frame(index)
            if not self.test_mode and skip == 1:
                # reload the previous valid frame
                # if no person in gt for train set
                index -= 1
            else:
                break

        k_tensor = k_tensor.double()
        r_tensor = r_tensor.double()
        t_tensor = t_tensor.double()

        dist_coeff_tensor = self.get_dist_coeff(index)
        height, width = self.get_resolution(index)[0]

        c = np.array([width / 2.0, height / 2.0])
        s = get_scale(np.array([width, height]), self.image_size)
        r = 0  # NOTE: do not apply rotation augmentation
        hm_scale = self.heatmap_size / self.image_size

        # Affine transformations
        trans, inv_trans, aug_trans = self.get_affine_transforms(
            c, s, hm_scale, r)
        kw_data['affine_trans'] = trans
        kw_data['inv_affine_trans'] = inv_trans
        kw_data['aug_trans'] = aug_trans
        kw_data['center'] = c
        kw_data['scale'] = s

        kw_data['n_person'] = n_person

        scene_idx, frame_idx, _ = self.process_index_mapping(index)

        if n_person > self.maximum_person:
            self.logger.error(
                f'{index}:{scene_idx}:{frame_idx}-{self.n_person}-'
                'Too many persons, please adjuest n_max_person')
            raise ValueError

        for view_idx in range(self.n_views):
            view_kw_data = copy.deepcopy(kw_data)

            # Camera parameters
            view_kw_data['camera'] = dict()
            if 'panoptic' in self.dataset:
                trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                             [0.0, 1.0, 0.0]]).double()
            elif 'shelf' or 'campus' in self.dataset:
                trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]]).double()
            else:
                self.logger.error('This dataset is not yet supported.')
                raise NotImplementedError

            r_trans = torch.mm(r_tensor[view_idx], trans_ground)
            t_trans = -torch.mm(r_trans.T, t_tensor[view_idx].reshape((3, 1)))
            dist_coeff = dist_coeff_tensor[view_idx]

            view_kw_data['camera']['camera_standard_T'] = t_tensor[view_idx]
            view_kw_data['camera']['R'] = r_trans
            view_kw_data['camera']['T'] = t_trans
            view_kw_data['camera']['K'] = k_tensor[view_idx]
            view_kw_data['camera']['dist_coeff'] = dist_coeff

            # GT keypoints
            kps3d_u = torch.zeros((self.maximum_person, self.n_kps, 3))
            kps3d_mask_u = torch.zeros((self.maximum_person, self.n_kps, 1))
            kps2d_u = torch.zeros((self.maximum_person, self.n_kps, 2))
            kps2d_mask_u = torch.zeros((self.maximum_person, self.n_kps, 1))

            for i, person_kps3d in enumerate(kps3d):
                person_kps3d = person_kps3d[:self.n_kps]  # [n_kps, 4]

                person_vis = person_kps3d[:, -1] > self.kps_thr

                if self.root_id is not None and not person_vis[self.root_id]:
                    continue

                trans_person_kps3d = torch.mm(person_kps3d[:, 0:3],
                                              trans_ground)
                kps3d_u[i] = trans_person_kps3d
                kps3d_mask_u[i] = torch.unsqueeze(person_vis, 1)

                kps2d = project_3dpts(trans_person_kps3d[:, 0:3].T,
                                      k_tensor[view_idx], r_trans,
                                      t_tensor[view_idx].reshape((3, 1)),
                                      dist_coeff.reshape((8, 1))).T[:, :2]
                kps2d_u[i] = affine_transform_pts(
                    kps2d, torch.tensor(trans[None, None, 0:2, :]))

                x_check = np.bitwise_and(kps2d_u[i][:, 0] >= 0,
                                         kps2d_u[i][:, 0] <= width - 1)
                y_check = np.bitwise_and(kps2d_u[i][:, 1] >= 0,
                                         kps2d_u[i][:, 1] <= height - 1)
                check = np.bitwise_and(x_check, y_check)
                person_vis[np.logical_not(check)] = 0

                kps2d_mask_u[i] = torch.unsqueeze(person_vis, 1)

            view_kw_data['kps3d'] = kps3d_u
            view_kw_data['kps3d_vis'] = kps3d_mask_u
            view_kw_data['kps2d'] = kps2d_u
            view_kw_data['kps2d_vis'] = kps2d_mask_u

            meta.append(view_kw_data)

        image = [img for img in mview_img_tensor]

        return image, meta

    def __len__(self):
        return super().__len__()

    def get_one_frame(self, index):
        skip = 0
        mview_img_tensor, k_tensor, r_tensor, t_tensor, kps3d, \
            _, kw_data = super().__getitem__(index)

        check_valid = torch.sum(kps3d, axis=1)  # [n_person, 4]
        kps3d = kps3d[check_valid[:, -1] > 0]

        n_person = kps3d.shape[0]
        if n_person == 0:  # skip the frame if no person
            skip = 1
        return mview_img_tensor, k_tensor, r_tensor, t_tensor, \
            kps3d, kw_data, n_person, skip

    def get_affine_transforms(self,
                              c: np.ndarray,
                              s: np.ndarray,
                              aug_s: np.ndarray,
                              r: int = 0
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get affine transformation matrix, inverse affine transformation
        matrix and augmented affine transformation with given.

        Args:
            c (np.ndarray): Center of the image.
            s (np.ndarray): Scale for affine transformation.
            aug_s (np.ndarray): Scale for augmented affine transformation.
            r (int, optional): Rotation. Defaults to 0.

        Returns:
            aff_trans (np.ndarray): Affine transformation matrix.
            inv_aff_trans (np.ndarray): Inverse affine transformation matrix.
            aug_trans (np.ndarray): Augmented affine transformation matrix.
        """
        aff_trans = np.eye(3, 3)
        inv_aff_trans = np.eye(3, 3)
        aug_trans = np.eye(3, 3)
        scale_trans = np.eye(3, 3)

        trans = get_affine_transform(c, s, r, self.image_size, inv=0)
        inv_trans = get_affine_transform(c, s, r, self.image_size, inv=1)

        aff_trans[0:2] = trans.copy()
        inv_aff_trans[0:2] = inv_trans.copy()
        aug_trans[0:2] = trans.copy()

        scale_trans[0, 0] = aug_s[1]
        scale_trans[1, 1] = aug_s[0]
        aug_trans = scale_trans.dot(aug_trans)

        return aff_trans, inv_aff_trans, aug_trans

    def get_dist_coeff(self, index: int) -> torch.tensor:
        """Get distortion coefficients.

        Args:
            index (int): Index in dataset.

        Returns:
            torch.tensor:
                Distorsion coeffeicient in shape (n_views, 8).
                The sequence is:
                ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        """
        dist_coeff_list = []
        scene_idx, _, _ = self.process_index_mapping(index)
        for fisheye_param in self.fisheye_params[scene_idx]:
            dist_coeff_list.append(
                torch.tensor(fisheye_param.get_dist_coeff()))

        dist_coeff_tensor = torch.stack(dist_coeff_list)
        return dist_coeff_tensor

    def get_resolution(self, index: int) -> torch.tensor:
        """Get image resolution.

        Args:
            index (int): Index in dataset.

        Returns:
            torch.tensor:
                Distorsion resolution in shape (n_views, 2).
                The sequence is: [height, width]
        """
        resolution = []
        scene_idx, _, _ = self.process_index_mapping(index)
        for fisheye_param in self.fisheye_params[scene_idx]:
            resolution.append(
                torch.tensor([fisheye_param.height, fisheye_param.width]))

        resolution = torch.stack(resolution)
        return resolution
