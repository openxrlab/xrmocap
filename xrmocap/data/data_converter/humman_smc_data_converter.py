# yapf: disable
import logging
import numpy as np
import os
from typing import Union
from xrprimer.data_structure import Keypoints
from xrprimer.transform.image.color import rgb2bgr
from xrprimer.utils.ffmpeg_utils import array_to_images
from xrprimer.utils.path_utils import (
    Existence, check_path_existence, check_path_suffix,
)

from xrmocap.data.data_visualization import MviewMpersonDataVisualization
from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_perception.builder import MMtrackDetector, build_detector
from xrmocap.io.camera import get_all_color_kinect_parameter_from_smc
from .base_data_converter import BaseDataCovnerter

# yapf: enable


class HummanSMCDataCovnerter(BaseDataCovnerter):

    def __init__(self,
                 data_root: str,
                 bbox_detector: Union[dict, None],
                 kps2d_estimator: Union[dict, None],
                 batch_size: int = 500,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'humman_smc',
                 visualize: bool = False,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Create a dir at meta_path, and convert camera parameters and
        detected bbox2d, kps2d into XRLab format.

        Args:
            data_root (str):
                Path to the directory with smc files.
            bbox_detector (Union[dict, None]):
                A human bbox_detector, or its config, or None.
                If None, converting perception 2d will be skipped.
            kps2d_estimator (Union[dict, None]):
                A top-down kps2d estimator, or its config, or None.
                If None, converting perception 2d will be skipped.
            batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset.
                Defaults to 'humman_smc'.
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
        self.batch_size = batch_size
        # check how many smc files in the data_root
        file_names = sorted(os.listdir(data_root))
        self.smc_paths = []
        for file_name in file_names:
            file_path = os.path.join(data_root, file_name)
            if check_path_suffix(file_path, '.smc'):
                self.smc_paths.append(file_path)
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
        for scene_idx, smc_path in enumerate(self.smc_paths):
            smc_reader = SMCReader(file_path=smc_path)
            scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
            os.makedirs(scene_dir, exist_ok=True)
            self.covnert_smc_frames(scene_idx, smc_reader)
            self.covnert_image_list(scene_idx, smc_reader)
            self.convert_cameras(scene_idx, smc_reader)
            if self.bbox_detector is not None and \
                    self.kps2d_estimator is not None:
                self.convert_perception_2d(scene_idx, smc_reader)
        if self.visualize:
            self.visualization.run(overwrite=overwrite)

    def covnert_smc_frames(self, scene_idx: int,
                           smc_reader: SMCReader) -> None:
        """Convert smc in source dataset to extracted jpg pictures.

        Args:
            scene_idx (int):
                Index of this scene.
            smc_reader (SMCReader):
                SMCReader of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        smc_path = self.smc_paths[scene_idx]
        smc_name = os.path.basename(smc_path)
        if self.verbose:
            self.logger.info('Extracting image relative path list' +
                             f' for scene {scene_idx}, file {smc_name}.')
        n_view = smc_reader.num_kinects
        for view_idx in range(n_view):
            output_folder = os.path.join(scene_dir,
                                         f'images_view_{view_idx:02d}')
            sv_img_array = smc_reader.get_kinect_color(kinect_id=view_idx)
            sv_img_array = rgb2bgr(sv_img_array)
            array_to_images(
                image_array=sv_img_array,
                output_folder=output_folder,
                img_format='%06d.jpg',
                logger=self.logger)

    def covnert_image_list(self, scene_idx: int,
                           smc_reader: SMCReader) -> None:
        """Convert source data to image list text file.

        Args:
            scene_idx (int):
                Index of this scene.
            smc_reader (SMCReader):
                SMCReader of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        smc_path = self.smc_paths[scene_idx]
        smc_name = os.path.basename(smc_path)
        if self.verbose:
            self.logger.info('Converting image relative path list' +
                             f' for scene {scene_idx}, file {smc_name}.')
        n_view = smc_reader.num_kinects
        n_frame = smc_reader.kinect_num_frames
        for view_idx in range(n_view):
            image_dir = os.path.join(scene_dir, f'images_view_{view_idx:02d}')
            frame_list = []
            for frame_idx in range(n_frame):
                file_name = f'{frame_idx:06d}.jpg'
                abs_file_path = os.path.join(image_dir, file_name)
                rela_file_path = os.path.relpath(abs_file_path, self.data_root)
                if check_path_existence(abs_file_path,
                                        'file') == Existence.FileExist:
                    frame_list.append(rela_file_path + '\n')
                else:
                    self.logger.error('Humman dataset is broken.' +
                                      f' Missing file: {abs_file_path}')
                    raise FileNotFoundError
            frame_list[-1] = frame_list[-1][:-1]
            with open(
                    os.path.join(scene_dir,
                                 f'image_list_view_{view_idx:02d}.txt'),
                    'w') as f_write:
                f_write.writelines(frame_list)

    def convert_cameras(self, scene_idx: int, smc_reader: SMCReader) -> None:
        """Convert smc data to XRPrimer camera parameters.

        Args:
            scene_idx (int):
                Index of this scene.
            smc_reader (SMCReader):
                SMCReader of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        smc_path = self.smc_paths[scene_idx]
        smc_name = os.path.basename(smc_path)
        if self.verbose:
            self.logger.info('Converting camera parameters' +
                             f' for scene {scene_idx}, file {smc_name}.')
        cam_param_list = get_all_color_kinect_parameter_from_smc(
            smc_reader=smc_reader, align_floor=True, logger=self.logger)
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        os.makedirs(cam_dir, exist_ok=True)
        for view_idx, cam_param in enumerate(cam_param_list):
            cam_param.name = f'fisheye_param_{view_idx:02d}'
            cam_param.dump(os.path.join(cam_dir, f'{cam_param.name}.json'))

    def convert_ground_truth(self, scene_idx: int,
                             smc_reader: SMCReader) -> None:
        """SMC has no GT inside, create an empty array.

        Args:
            scene_idx (int):
                Index of this scene.
            smc_reader (SMCReader):
                SMCReader of this scene.
        """
        # DOING
        if self.verbose:
            self.logger.info(
                f'Converting ground truth keypoints3d for scene {scene_idx}.')
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        n_frame = smc_reader.kinect_num_frames
        kps3d_arr = np.zeros(shape=(n_frame, 1, 17, 4))
        kps3d_mask = np.zeros_like(kps3d_arr[..., -1:])
        keypoints3d = Keypoints(
            kps=kps3d_arr,
            mask=kps3d_mask,
            convention='coco',
            logger=self.logger)
        keypoints3d.dump(os.path.join(scene_dir, 'keypoints3d_GT.npz'))

    def convert_perception_2d(self, scene_idx: int,
                              smc_reader: SMCReader) -> None:
        """Convert source data to 2D perception data, bbox + keypoints2d +
        track info.

        Args:
            scene_idx (int):
                Index of this scene.
            smc_reader (SMCReader):
                SMCReader of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        scene_bbox_list = []
        scene_keypoints_list = []
        n_view = smc_reader.num_kinects
        for view_idx in range(n_view):
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
