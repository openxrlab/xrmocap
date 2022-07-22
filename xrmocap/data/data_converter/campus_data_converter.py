# yapf: disable
import logging
import numpy as np
import os
from scipy.io import loadmat
from typing import List, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data.data_visualization import MviewMpersonDataVisualization
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.human_detection.builder import MMtrackDetector, build_detector
from xrmocap.transform.convention.keypoints_convention import get_keypoint_num
from xrmocap.utils.path_utils import Existence, check_path_existence
from .base_data_converter import BaseDataCovnerter

# yapf: enable


class CampusDataCovnerter(BaseDataCovnerter):

    def __init__(self,
                 data_root: str,
                 bbox_detector: Union[dict, None] = None,
                 kps2d_estimator: Union[dict, None] = None,
                 scene_range: List[List[int]] = None,
                 batch_size: int = 500,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'campus',
                 visualize: bool = False,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Create a dir at meta_path, split data into scenes according to
        scene_range, and convert GT kps3d, camera parameters and detected
        bbox2d, kps2d into XRLab format.

        Args:
            data_root (str):
                Path to the Campus dataset.
                Typically it is the path to CampusSeq1.
            bbox_detector (Union[dict, None]):
                A human bbox_detector, or its config, or None.
                If None, converting perception 2d will be skipped.
                Defaults to None.
            kps2d_estimator (Union[dict, None]):
                A top-down kps2d estimator, or its config, or None.
                If None, converting perception 2d will be skipped.
                Defaults to None.
            scene_range (List[List[int]], optional):
                Frame range of scenes. For instance, [[350, 470], [650, 750]]
                will split the dataset into 2 scenes,
                scene_0: 350-470, scene_1 650-750.
                Defaults to None, scene_0: 0-2000.
            batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset.
                Defaults to 'campus'.
            visualize (bool, optional):
                Whether to visualize perception2d data and
                ground-truth 3d data. Defaults to False.
            verbose (bool, optional):
                Whether to print(logger.info) information during converting.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(
            data_root=data_root,
            meta_path=meta_path,
            dataset_name=dataset_name,
            verbose=verbose,
            logger=logger)
        self.n_view = 3
        # index: scene idx, value: (start_frame_idx, end_frame_idx)
        if scene_range is not None:
            self.scene_range = scene_range
        else:
            self.scene_range = [
                [0, 2000],
            ]
        self.batch_size = batch_size
        if isinstance(bbox_detector, dict):
            bbox_detector['logger'] = logger
            self.bbox_detector = build_detector(bbox_detector)
        else:
            self.bbox_detector = bbox_detector

        if isinstance(kps2d_estimator, dict):
            kps2d_estimator['logger'] = logger
            self.kps2d_estimator = build_detector(kps2d_estimator)
        else:
            self.kps2d_estimator = kps2d_estimator
        self.visualize = visualize
        if self.visualize:
            vis_percep2d = self.bbox_detector is not None and \
                    self.kps2d_estimator is not None
            self.visualization = MviewMpersonDataVisualization(
                data_root=data_root,
                meta_path=meta_path,
                vis_percep2d=vis_percep2d,
                output_dir=os.path.join(meta_path, 'visualize'),
                verbose=verbose,
                logger=logger)

    def run(self, overwrite: bool = False) -> None:
        """Convert xrmocap meta data from source dataset. Scenes will be
        created, and there are image lists, XRPrimer cameras, ground truth
        keypoints3d, perception2d in each of the scene. If any of
        bbox_detector, kps2d_estimator is None, skip converting perception2d.
        If visualize is checked, perception2d and ground truth will be
        visualized.

        Args:
            overwrite (bool, optional):
                Whether replace the files at
                self.meta_path.
                Defaults to False.
        """
        BaseDataCovnerter.run(self, overwrite=overwrite)
        with open(os.path.join(self.meta_path, 'dataset_name.txt'),
                  'w') as f_write:
            f_write.write(f'{self.dataset_name}')
        for scene_idx in range(len(self.scene_range)):
            scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
            os.makedirs(scene_dir, exist_ok=True)
            self.covnert_image_list(scene_idx)
            self.convert_cameras(scene_idx)
            self.convert_ground_truth(scene_idx)
            if self.bbox_detector is not None and \
                    self.kps2d_estimator is not None:
                self.convert_perception_2d(scene_idx)
        if self.visualize:
            self.visualization.run(overwrite=overwrite)

    def covnert_image_list(self, scene_idx: int) -> None:
        """Convert source data to image list text file.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info('Converting image relative path list' +
                             f' for scene {scene_idx}.')
        for view_idx in range(self.n_view):
            frame_dir = os.path.join(self.data_root, f'Camera{view_idx}')
            frame_list = []
            start_idx = self.scene_range[scene_idx][0]
            end_idx = self.scene_range[scene_idx][1]
            for frame_idx in range(start_idx, end_idx):
                file_name = f'campus4-c{view_idx}-{frame_idx:05d}.png'
                rela_file_path = os.path.join(f'Camera{view_idx}', file_name)
                abs_file_path = os.path.join(frame_dir, file_name)
                if check_path_existence(abs_file_path,
                                        'file') == Existence.FileExist:
                    frame_list.append(rela_file_path + '\n')
                else:
                    self.logger.error('Campus dataset is broken.' +
                                      f' Missing file: {abs_file_path}')
                    raise FileNotFoundError
            frame_list[-1] = frame_list[-1][:-1]
            with open(
                    os.path.join(scene_dir,
                                 f'image_list_view_{view_idx:02d}.txt'),
                    'w') as f_write:
                f_write.writelines(frame_list)

    def convert_cameras(self, scene_idx: int) -> None:
        """Convert source data to XRPrimer camera parameters. Intrinsic in
        Calibration/producePmat.m can be trusted while extrinsic cannot. We get
        extrinsic by:

        K^(-1) * K * [ R | T ]

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info(
                f'Converting camera parameters for scene {scene_idx}.')
        cam_param_path = os.path.join(self.data_root, 'Calibration',
                                      'producePmat.m')
        with open(cam_param_path, 'r') as f_read:
            lines = f_read.readlines()
        param_keys = ['width', 'height', 'scale', 'focal', 'cx', 'cy', 'sx']
        param_dict = {}
        for key in param_keys:
            param_dict[key] = []
        for line in lines:
            if '=' in line:
                line = line.strip().replace(';', '')
                strs = line.split('=')
                if strs[0] in param_dict:
                    param_dict[strs[0]].append(float(strs[1]))
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        os.makedirs(cam_dir, exist_ok=True)
        scale = param_dict['scale'][0]
        width = int(param_dict['width'][0] * scale)
        height = int(param_dict['height'][0] * scale)
        for view_idx in range(self.n_view):
            fisheye_param = FisheyeCameraParameter(
                name=f'fisheye_param_{view_idx:02d}')
            fisheye_param.set_intrinsic(
                width=width,
                height=height,
                fx=param_dict['focal'][view_idx] / param_dict['sx'][view_idx] *
                scale,
                fy=param_dict['focal'][view_idx] / param_dict['sx'][view_idx] *
                scale,
                cx=param_dict['cx'][view_idx] * scale,
                cy=param_dict['cy'][view_idx] * scale)
            proj_mat_path = os.path.join(self.data_root, 'Calibration',
                                         f'P{view_idx}.txt')
            with open(proj_mat_path, 'r') as f_read:
                lines = f_read.readlines()
            one_line = (lines[0] + lines[1] + lines[2]).replace('\n', ' ')
            strs = one_line.split(' ')[:12]
            proj_mat = np.asarray([float(str_value)
                                   for str_value in strs]).reshape(3, 4)
            rt_mat = np.matmul(
                np.linalg.inv(fisheye_param.get_intrinsic(3)), proj_mat)
            fisheye_param.set_KRT(
                R=rt_mat[:, :3], T=rt_mat[:, 3], world2cam=True)
            fisheye_param.dump(
                os.path.join(cam_dir, f'{fisheye_param.name}.json'))

    def convert_ground_truth(self, scene_idx: int) -> None:
        """Convert source data to keypoints3d GT data.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        if self.verbose:
            self.logger.info(
                f'Converting ground truth keypoints3d for scene {scene_idx}.')
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        start_idx = self.scene_range[scene_idx][0]
        end_idx = self.scene_range[scene_idx][1]
        gt_path = os.path.join(self.data_root, 'actorsGT.mat')
        gt_mat = loadmat(gt_path)
        kps3d_mat = gt_mat['actor3D'][0]
        n_person = len(kps3d_mat)
        n_frame = end_idx - start_idx
        n_kps = get_keypoint_num('campus')
        kps3d_arr = np.zeros(shape=(n_frame, n_person, n_kps, 3))
        kps3d_mask = np.zeros(shape=(n_frame, n_person, n_kps), dtype=np.uint8)
        for p_idx in range(n_person):
            for f_idx in range(n_frame):
                sview_sperson_mat = kps3d_mat[p_idx][f_idx + start_idx][0]
                if len(sview_sperson_mat) == 1:
                    kps3d_mask[f_idx, p_idx, :] = 0
                else:
                    kps3d_mask[f_idx, p_idx, :] = 1
                    kps3d_arr[f_idx, p_idx, :, :] = sview_sperson_mat
        kps3d_conf = np.expand_dims(
            kps3d_mask.astype(kps3d_arr.dtype), axis=-1)
        kps3d_arr = np.concatenate((kps3d_arr, kps3d_conf), axis=-1)
        keypoints3d = Keypoints(
            kps=kps3d_arr,
            mask=kps3d_mask,
            convention='campus',
            logger=self.logger)
        keypoints3d.dump(os.path.join(scene_dir, 'keypoints3d_GT.npz'))

    def convert_perception_2d(self, scene_idx: int) -> None:
        """Convert source data to 2D perception data, bbox + keypoints2d +
        track info.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        scene_bbox_list = []
        scene_keypoints_list = []
        for view_idx in range(self.n_view):
            if self.verbose:
                self.logger.info('Inferring perception 2D data for' +
                                 f' scene {scene_idx} view {view_idx}')
            list_path = os.path.join(scene_dir,
                                     f'image_list_view_{view_idx:02d}.txt')
            with open(list_path, 'r') as f_read:
                rela_path_list = f_read.readlines()
            frame_list = [
                os.path.join(self.data_root, rela_path.strip())
                for rela_path in rela_path_list
            ]
            bbox_list = self.bbox_detector.infer_frames(
                frame_path_list=frame_list,
                disable_tqdm=not self.verbose,
                multi_person=True,
                load_batch_size=self.batch_size)
            kps2d_list, _, _ = self.kps2d_estimator.infer_frames(
                frame_path_list=frame_list,
                bbox_list=bbox_list,
                disable_tqdm=not self.verbose,
                load_batch_size=self.batch_size)
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            n_person_max = 0
            n_frame = len(bbox_list)
            for frame_idx, frame_bboxes in enumerate(bbox_list):
                if n_person_max < len(frame_bboxes):
                    n_person_max = len(frame_bboxes)
            bbox_arr = np.zeros(shape=(n_frame, n_person_max, 5))
            for frame_idx, frame_bboxes in enumerate(bbox_list):
                for person_idx, person_bbox in enumerate(frame_bboxes):
                    if person_bbox is not None:
                        bbox_arr[frame_idx, person_idx] = person_bbox
            scene_bbox_list.append(bbox_arr)
            scene_keypoints_list.append(keypoints2d)
        dict_to_save = dict(
            bbox_tracked=isinstance(self.bbox_detector, MMtrackDetector),
            bbox_convention='xyxy',
            kps2d_convention=keypoints2d.get_convention(),
        )
        for view_idx in range(self.n_view):
            dict_to_save[f'bbox2d_view_{view_idx:02d}'] = scene_bbox_list[
                view_idx]
            dict_to_save[f'kps2d_view_{view_idx:02d}'] = scene_keypoints_list[
                view_idx].get_keypoints()
            dict_to_save[
                f'kps2d_mask_view_{view_idx:02d}'] = scene_keypoints_list[
                    view_idx].get_mask()
        np.savez_compressed(
            file=os.path.join(scene_dir, 'perception_2d.npz'), **dict_to_save)
