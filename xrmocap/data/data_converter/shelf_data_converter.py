# yapf: disable
import logging
import numpy as np
import os
from typing import List, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.convention.camera import convert_camera_parameter

from xrmocap.utils.path_utils import Existence, check_path_existence
from .campus_data_converter import CampusDataCovnerter

# yapf: enable


class ShelfDataCovnerter(CampusDataCovnerter):

    def __init__(self,
                 data_root: str,
                 bbox_detector: Union[dict, None] = None,
                 kps2d_estimator: Union[dict, None] = None,
                 scene_range: List[List[int]] = 'all',
                 gt_person_idxs: List[int] = [0, 1, 2],
                 batch_size: int = 500,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'shelf',
                 visualize: bool = False,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Create a dir at meta_path, split data into scenes according to
        scene_range, and convert GT kps3d, camera parameters and detected
        bbox2d, kps2d into XRLab format.

        Args:
            data_root (str):
                Path to the Shelf dataset.
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
                Defaults to None, scene_0: 0-3200.
            gt_person_idxs (List[int], optional):
                A list of person indexes. Ground truth of the selected people
                will be converted, as mvpose only evaluates 3/4 of Shelf GT.
                Defaults to [0, 1, 2].
            batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset.
                Defaults to 'shelf'.
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
        if scene_range == 'all':
            self.scene_range = [
                [0, 3200],
            ]
        super().__init__(
            data_root=data_root,
            meta_path=meta_path,
            dataset_name=dataset_name,
            verbose=verbose,
            logger=logger,
            scene_range=scene_range,
            bbox_detector=bbox_detector,
            kps2d_estimator=kps2d_estimator,
            gt_person_idxs=gt_person_idxs,
            visualize=visualize,
            batch_size=batch_size)
        self.n_view = 5

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
                file_name = f'img_{frame_idx:06d}.png'
                rela_file_path = os.path.join(f'Camera{view_idx}', file_name)
                abs_file_path = os.path.join(frame_dir, file_name)
                if check_path_existence(abs_file_path,
                                        'file') == Existence.FileExist:
                    frame_list.append(rela_file_path + '\n')
                else:
                    self.logger.error(
                        f'{self.dataset_name} dataset is broken.' +
                        f' Missing file: {abs_file_path}')
                    raise FileNotFoundError
            frame_list[-1] = frame_list[-1][:-1]
            with open(
                    os.path.join(scene_dir,
                                 f'image_list_view_{view_idx:02d}.txt'),
                    'w') as f_write:
                f_write.writelines(frame_list)

    def convert_cameras(self, scene_idx: int) -> None:
        """Convert source data to XRPrimer camera parameters.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        if self.verbose:
            self.logger.info(
                f'Converting camera parameters for scene {scene_idx}.')
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        os.makedirs(cam_dir, exist_ok=True)
        for view_idx in range(self.n_view):
            fisheye_param = FisheyeCameraParameter(
                name=f'fisheye_param_{view_idx:02d}', convention='opencv')
            cam_param_path = os.path.join(self.data_root, 'Calibration',
                                          f'Camera{view_idx}.cal')
            with open(cam_param_path, 'r') as f_read:
                lines = f_read.readlines()
            resolution = __parse_mat_from_lines__(
                lines=lines[3:4], shape=[1, 2])
            width = int(resolution[0, 0])
            height = int(resolution[0, 1])
            # Shelf camera is not defined in opencv convention
            # rotate R, T to opencv
            rot_mat = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ])
            intrinsic33 = __parse_mat_from_lines__(
                lines=lines[4:7], shape=[3, 3])
            extrinsic_r = __parse_mat_from_lines__(
                lines=lines[7:10], shape=[3, 3])
            extrinsic_r = np.matmul(rot_mat, extrinsic_r)
            extrinsic_t = __parse_mat_from_lines__(
                lines=lines[10:11], shape=[1, 3]).reshape(3)
            extrinsic_t = np.matmul(extrinsic_t, rot_mat).reshape(3)
            fisheye_param.set_resolution(height=height, width=width)
            fisheye_param.set_KRT(
                K=intrinsic33, R=extrinsic_r, T=extrinsic_t, world2cam=True)
            opencv_fisheye_param = convert_camera_parameter(
                fisheye_param, 'opencv')
            opencv_fisheye_param.dump(
                os.path.join(cam_dir, f'{opencv_fisheye_param.name}.json'))


def __parse_mat_from_lines__(lines: List[str], shape: List[int]) -> np.ndarray:
    ret_arr = np.zeros(shape=shape)
    for row_idx in range(shape[0]):
        param_strs = lines[row_idx].split(' ')
        for col_idx in range(shape[1]):
            ret_arr[row_idx, col_idx] = float(param_strs[col_idx].strip())
    return ret_arr
