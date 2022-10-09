# yapf: disable
import logging
import numpy as np
import os
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.core.visualization import (
    visualize_keypoints2d, visualize_project_keypoints3d,
)
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from xrmocap.utils.ffmpeg_utils import mview_array_to_video
from .base_data_visualization import BaseDataVisualization

# yapf: enable


class MviewMpersonDataVisualization(BaseDataVisualization):

    def __init__(self,
                 data_root: str,
                 output_dir: str,
                 bbox_thr: float = 0.0,
                 vis_percep2d: bool = True,
                 kps2d_convention: Union[None, str] = None,
                 pred_kps3d_paths: List[str] = None,
                 pred_kps3d_convention: Union[None, str] = None,
                 vis_gt_kps3d: bool = True,
                 vis_bottom_up: bool = False,
                 resolution: Tuple = None,
                 gt_kps3d_convention: Union[None, str] = None,
                 vis_cameras: bool = False,
                 vis_aio_video: bool = True,
                 meta_path: str = 'xrmocap_meta',
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Visualize converted GT kps3d, camera parameters and detected bbox2d,
        kps2d from a MviewMpersonDataset.

        Args:
            data_root (str):
                Path to the multi-view multi-person dataset.
            output_dir (str):
                Path to the directory for all visualized videos.
            bbox_thr (float, optional):
                Threshold of bbox. If someone's bbox score is
                lower than bbox_thr,
                its bbox and kps will be set to zeros, won't
                be seen. Defaults to 0.0.
            vis_percep2d (bool, optional):
                Whether to visualize perception2d data.
                Defaults to True.
            kps2d_convention (Union[None, str], optional):
                Target convention of keypoints2d, if None,
                kps2d will keep its convention in meta-data.
                Defaults to None.
            pred_kps3d_paths (List[str], optional):
                Paths to the predicted keypoints3d npz files.
                Assume there's 2 scenes, you can pass either
                [path0, path1] or [path0, ''].
                Defaults to None.
            pred_kps3d_convention (Union[None, str], optional):
                Target convention of predicted keypoints3d,
                if None,
                kps3d will keep its convention in meta-data.
                Defaults to None.
            vis_gt_kps3d (bool, optional):
                Whether to visualize groundtruth3d data.
                Defaults to True.
            gt_kps3d_convention (Union[None, str], optional):
                Target convention of groundtruth keypoints3d,
                if None,
                kps3d will keep its convention in meta-data.
                Defaults to None.
            vis_cameras (bool, optional):
                Whether to visualize cameras in the scene.
                Defaults to False.
            vis_aio_video (bool, optional):
                Whether to concat videos from all views together
                and output an all-in-one video.
                Defaults to True.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
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
            output_dir=output_dir,
            verbose=verbose,
            logger=logger)
        self.bbox_thr = bbox_thr
        self.vis_percep2d = vis_percep2d
        self.kps2d_convention = kps2d_convention
        self.vis_gt_kps3d = vis_gt_kps3d
        self.vis_bottom_up = vis_bottom_up
        self.gt_kps3d_convention = gt_kps3d_convention
        self.vis_cameras = vis_cameras
        self.vis_aio_video = vis_aio_video
        self.pred_kps3d_paths = pred_kps3d_paths \
            if pred_kps3d_paths is not None \
            else []
        self.pred_kps3d_convention = pred_kps3d_convention
        self.resolution = resolution

    def run(self, overwrite: bool = False) -> None:
        """Visualize meta-data selected in __init__().

        Args:
            overwrite (bool, optional):
                Whether replace the files at
                self.output_dir.
                Defaults to False.
        """
        super().run(overwrite=overwrite)
        file_names = sorted(os.listdir(self.meta_path))
        scene_names = []
        for file_name in file_names:
            if file_name.startswith('scene_'):
                scene_names.append(file_name)
        if len(self.pred_kps3d_paths) != 0 and \
                len(self.pred_kps3d_paths) != len(scene_names):
            self.logger.error(
                f'There are {len(scene_names)} scenes found at meta_path,' +
                ' but length of pred_kps3d_paths' +
                f' is {len(self.pred_kps3d_paths)}.'
                ' Please align' +
                ' pred_kps3d_paths\' length like [\'\', path, \'\']')
        for scene_idx, scene_name in enumerate(scene_names):
            scene_vis_dir = os.path.join(self.output_dir, f'scene_{scene_idx}')
            os.makedirs(scene_vis_dir, exist_ok=True)
            if len(self.pred_kps3d_paths) > 0:
                self.visualize_predict_3d(scene_idx)
            if self.vis_percep2d:
                self.visualize_perception_2d(scene_idx)
            if self.vis_gt_kps3d:
                self.visualize_ground_truth_3d(scene_idx)
            if self.vis_bottom_up:
                self.visualize_perception_2d_bottm_up(scene_idx)

    def visualize_perception_2d(self, scene_idx: int) -> None:
        """Visualize converted 2D perception keypoints2d data. If bbox was
        tracked, tracking relationship will be represented by color of
        keypoints.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        perception2d_path = os.path.join(scene_dir, 'perception_2d.npz')
        perception2d_dict = np.load(perception2d_path, allow_pickle=True)
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        file_names = sorted(os.listdir(cam_dir))
        cam_names = []
        view_idxs = []
        for file_name in file_names:
            if file_name.startswith('fisheye_param_'):
                cam_names.append(file_name)
                view_idxs.append(
                    int(
                        file_name.replace('fisheye_param_',
                                          '').replace('.json', '')))
        n_view = len(cam_names)
        mview_plot_arr = []
        for idx in range(n_view):
            view_idx = view_idxs[idx]
            if self.verbose:
                self.logger.info('Visualizing perception 2D data for' +
                                 f' scene {scene_idx} view {view_idx}')
            list_path = os.path.join(scene_dir,
                                     f'image_list_view_{view_idx:02d}.txt')
            with open(list_path, 'r') as f_read:
                rela_path_list = f_read.readlines()
            frame_list = [
                os.path.join(self.data_root, rela_path.strip())
                for rela_path in rela_path_list
            ]
            bbox = perception2d_dict[f'bbox2d_view_{view_idx:02d}']
            kps2d = perception2d_dict[f'kps2d_view_{view_idx:02d}']
            kps2d_mask = perception2d_dict[f'kps2d_mask_view_{view_idx:02d}']
            kps2d_convention = perception2d_dict['kps2d_convention'].item()
            for frame_idx, frame_bboxes in enumerate(bbox):
                for person_idx, person_bbox in enumerate(frame_bboxes):
                    if person_bbox is not None and \
                            person_bbox[4] < self.bbox_thr:
                        kps2d_mask[frame_idx, person_idx, ...] = 0
                        kps2d[frame_idx, person_idx, ...] = 0
            keypoints2d = Keypoints(
                kps=kps2d, mask=kps2d_mask, convention=kps2d_convention)
            if self.kps2d_convention is not None:
                keypoints2d = convert_keypoints(
                    keypoints2d, dst=self.kps2d_convention, approximate=True)
            scene_vis_dir = os.path.join(self.output_dir, f'scene_{scene_idx}')
            video_path = os.path.join(scene_vis_dir,
                                      f'perception2d_view_{view_idx:02d}.mp4')
            plot_arr = visualize_keypoints2d(
                keypoints=keypoints2d,
                output_path=video_path,
                img_paths=frame_list,
                overwrite=True,
                resolution=self.resolution,
                return_array=self.vis_aio_video)
            mview_plot_arr.append(plot_arr)
        # draw views all in one
        if self.vis_aio_video:
            video_path = os.path.join(scene_vis_dir, 'perception2d_AIO.mp4')
            mview_array_to_video(
                mview_plot_arr, video_path, logger=self.logger)

    def visualize_perception_2d_bottm_up(self, scene_idx: int) -> None:
        """Visualize bottom-up associated 2D perception keypoints2d data.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        npz_path = os.path.join(self.output_dir,
                                f'scene{scene_idx}_associate_keypoints2d.npy')
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        file_names = sorted(os.listdir(cam_dir))
        cam_names = []
        view_idxs = []
        for file_name in file_names:
            if file_name.startswith('fisheye_param_'):
                cam_names.append(file_name)
                view_idxs.append(
                    int(
                        file_name.replace('fisheye_param_',
                                          '').replace('.json', '')))
        n_view = len(cam_names)
        mview_plot_arr = []
        arr_data = np.load(npz_path)
        for idx in range(n_view):
            view_idx = view_idxs[idx]
            if self.verbose:
                self.logger.info('Visualizing perception 2D data for' +
                                 f' scene {scene_idx} view {view_idx}')
            list_path = os.path.join(scene_dir,
                                     f'image_list_view_{view_idx:02d}.txt')
            with open(list_path, 'r') as f_read:
                rela_path_list = f_read.readlines()
            frame_list = [
                os.path.join(self.data_root, rela_path.strip())
                for rela_path in rela_path_list
            ]
            kps2d = arr_data[:, :, idx, :, :]
            keypoints2d = Keypoints(
                kps=kps2d,
                mask=kps2d[..., 2] > 0,
                convention=self.kps2d_convention)
            if self.kps2d_convention is not None:
                keypoints2d = convert_keypoints(
                    keypoints2d, dst=self.kps2d_convention, approximate=True)
            scene_vis_dir = os.path.join(self.output_dir, f'scene_{scene_idx}')
            video_path = os.path.join(
                scene_vis_dir, f'associate_kps2d_view_{view_idx:02d}.mp4')
            plot_arr = visualize_keypoints2d(
                keypoints=keypoints2d,
                output_path=video_path,
                img_paths=frame_list,
                resolution=self.resolution,
                overwrite=True,
                return_array=self.vis_aio_video)
            mview_plot_arr.append(plot_arr)
        # draw views all in one
        if self.vis_aio_video:
            video_path = os.path.join(scene_vis_dir, 'associate_kps2d_AIO.mp4')
            mview_array_to_video(
                mview_plot_arr, video_path, logger=self.logger)

    def visualize_ground_truth_3d(self, scene_idx: int) -> None:
        """Visualize converted ground truth keypoints3d data.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        keypoints3d = Keypoints.fromfile(
            os.path.join(scene_dir, 'keypoints3d_GT.npz'))
        if self.gt_kps3d_convention is not None:
            keypoints3d = convert_keypoints(
                keypoints3d, dst=self.gt_kps3d_convention, approximate=True)
        self.visualize_keypoints3d(scene_idx, keypoints3d, 'groundtruth3d')

    def visualize_predict_3d(self, scene_idx: int) -> None:
        """Visualize predicted keypoints3d data.

        Args:
            scene_idx (int):
                Index of this scene.
        """
        if len(self.pred_kps3d_paths) != 0:
            keypoints_path = self.pred_kps3d_paths[scene_idx]
            keypoints3d = Keypoints.fromfile(keypoints_path)
            if self.pred_kps3d_convention is not None:
                keypoints3d = convert_keypoints(
                    keypoints3d,
                    dst=self.pred_kps3d_convention,
                    approximate=True)
            self.visualize_keypoints3d(scene_idx, keypoints3d, 'predict3d')

    def visualize_keypoints3d(self,
                              scene_idx: int,
                              keypoints3d: Keypoints,
                              output_prefix: str = 'groundtruth3d') -> None:
        """Visualize keypoints3d data, ground truth or predicted results.

        Args:
            scene_idx (int):
                Index of this scene.
            keypoints3d (Keypoints):
                Keypoints3d to be projected.
            output_prefix (str, optional):
                Prefix of the output video files.
                Defaults to 'groundtruth3d'.
        """
        scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
        cam_dir = os.path.join(scene_dir, 'camera_parameters')
        file_names = sorted(os.listdir(cam_dir))
        cam_names = []
        view_idxs = []
        for file_name in file_names:
            if file_name.startswith('fisheye_param_'):
                cam_names.append(file_name)
                view_idxs.append(
                    int(
                        file_name.replace('fisheye_param_',
                                          '').replace('.json', '')))
        n_view = len(cam_names)
        mview_plot_arr = []
        for idx in range(n_view):
            view_idx = view_idxs[idx]
            if self.verbose:
                self.logger.info(f'Visualizing {output_prefix} data for' +
                                 f' scene {scene_idx} view {view_idx}')
            fisheye_param = FisheyeCameraParameter.fromfile(
                os.path.join(cam_dir, f'fisheye_param_{view_idx:02d}.json'))
            list_path = os.path.join(scene_dir,
                                     f'image_list_view_{view_idx:02d}.txt')
            with open(list_path, 'r') as f_read:
                rela_path_list = f_read.readlines()
            frame_list = [
                os.path.join(self.data_root, rela_path.strip())
                for rela_path in rela_path_list
            ]
            scene_vis_dir = os.path.join(self.output_dir, f'scene_{scene_idx}')
            video_path = os.path.join(
                scene_vis_dir,
                f'{output_prefix}_project_view_{view_idx:02d}.mp4')
            plot_arr = visualize_project_keypoints3d(
                keypoints=keypoints3d,
                cam_param=fisheye_param,
                output_path=video_path,
                img_paths=frame_list,
                overwrite=True,
                return_array=self.vis_aio_video)
            mview_plot_arr.append(plot_arr)
        # draw views all in one
        if self.vis_aio_video:
            video_path = os.path.join(scene_vis_dir,
                                      f'{output_prefix}_project_AIO.mp4')
            mview_array_to_video(
                mview_plot_arr, video_path, logger=self.logger)
