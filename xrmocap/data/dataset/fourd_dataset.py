import glob
import logging
import numpy as np
import os
import re
import torch
import pickle
import cv2
from typing import Tuple, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from .base_dataset import BaseDataset

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class FourDDataset(BaseDataset):

    def __init__(self,
                 data_root: str,
                 img_pipeline: list,
                 meta_path: str = 'xrmocap_meta',
                 test_mode: bool = True,
                 shuffled: bool = False,
                 metric_unit: Literal['meter', 'centimeter',
                                      'millimeter'] = 'meter',
                 bbox_convention: Union[None, Literal['xyxy', 'xywh']] = None,
                 bbox_thr: float = 0.0,
                 kps2d_convention: Union[None, str] = None,
                 gt_kps3d_convention: Union[None, str] = None,
                 cam_world2cam: bool = False,
                 cam_k_dim: int = 3,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """A dataset loads multi-view multi-person data from source dataset and
        meta-data from data converter.

        Args:
            data_root (str):
                Root path of the downloaded dataset.
            img_pipeline (list):
                A list of image transform instances.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            test_mode (bool, optional):
                Whether this dataset is used to load testset.
                Defaults to True.
            shuffled (bool, optional):
                Whether this dataset is used to load shuffled frames.
                If True, getitem will always get end_of_clip=True.
                Defaults to False.
            metric_unit (Literal[
                    'meter', 'centimeter', 'millimeter'], optional):
                Metric unit of gt3d and camera parameters. Defaults to 'meter'.
            bbox_convention (Union[None, Literal['xyxy', 'xywh']], optional):
                Target convention of bbox, if None,
                bbox will not be returned by getitem.
                Defaults to None.
            bbox_thr (float, optional):
                Threshold of bbox. If someone's bbox score is
                lower than bbox_thr,
                its bbox and kps will be set to zeros.
                Defaults to 0.0.
            kps2d_convention (Union[None, str], optional):
                Target convention of keypoints2d, if None,
                kps2d will not be returned by getitem. Defaults to None.
            gt_kps3d_convention (Union[None, str], optional):
                Target convention of keypoints3d, if None,
                kps3d will keep its convention in meta-data.
                Defaults to None.
            cam_world2cam (bool, optional):
                Direction of returned camera extrinsics.
                Defaults to False.
            cam_k_dim (int, optional):
                Dimension of returned camera intrinsic mat.
                Defaults to 3.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        with open(os.path.join(meta_path, 'dataset_name.txt'), 'r') as f_read:
            dataset_name = f_read.readline().strip()
        BaseDataset.__init__(
            self,
            data_root=data_root,
            img_pipeline=img_pipeline,
            meta_path=meta_path,
            dataset_name=dataset_name,
            test_mode=test_mode,
            logger=logger)
        # assign attr val
        self.shuffled = shuffled
        self.metric_unit = metric_unit
        if self.metric_unit == 'meter':
            self.factor = 1.0
        elif self.metric_unit == 'centimeter':
            self.factor = 100.0
        elif self.metric_unit == 'millimeter':
            self.factor = 1000.0
        else:
            self.logger.error(f'Wrong metric unit: {self.metric_unit}')
            raise ValueError
        self.bbox_convention = bbox_convention
        self.bbox_thr = bbox_thr
        self.kps2d_convention = kps2d_convention
        self.gt_kps3d_convention = gt_kps3d_convention
        self.cam_world2cam = cam_world2cam
        self.cam_k_dim = cam_k_dim
        # init empty attr
        self.index_mapping = []
        self.image_list = []
        self.fisheye_params = []
        self.view_idxs = []
        self.gt3d = []
        self.percep_bbox2d = []
        self.percep_keypoints2d = []
        # count scene
        scene_names = glob.glob(os.path.join(meta_path, 'scene_*'))
        self.n_scene = len(scene_names)
        # load meta-data from file
        self.load_camera_parameters()
        self.load_image_list()
        self.init_index_mapping()
        self.load_ground_truth_3d()

        self.load_perception_2d()        

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, bool, dict]:
        """Get item from the dataset.

        Args:
            index (int): Index in dataset.

        Returns:
            mview_img_tensor (torch.Tensor):
                Multi-view image tensor in shape (n_view, h, w, c).
            k_tensor (torch.Tensor):
                Multi-view intrinsic tensor in shape (n_view, 3, 3).
            r_tensor (torch.Tensor):
                Multi-view rotation mat tensor in shape (n_view, 3, 3).
            t_tensor (torch.Tensor):
                Multi-view translation vector tensor in shape (n_view, 3, 3).
            kps3d (torch.Tensor):
                Multi-view kps3d tensor in shape (n_person, n_kps, 4),
                while the last dim is confidence. If kps3d[p, k, 4] == 0,
                kps3d[p, k] is invalid and do not use the data.
            end_of_clip (bool):
                Whether it is the last frame of this clip. When shuffled,
                it is always False.
            kw_data (dict):
                Dict for keyword data. bbox and kps2d can be found here.
        """
        if index >= len(self):
            raise StopIteration
        scene_idx, frame_idx, end_of_clip = self.process_index_mapping(index)
        # load multi-view images
        img_paths = self.image_list[scene_idx][frame_idx]
        mview_img_list = []
        for rela_path in img_paths:
            img_path = os.path.join(self.data_root, rela_path)
            img_tensor = self.img_pipeline(img_path)
            mview_img_list.append(img_tensor)
        mview_img_tensor = torch.stack(mview_img_list)
        
        k_list = []
        r_list = []
        t_list = []
        # prepare multi-view cameras
        for fisheye_param in self.fisheye_params[scene_idx]:
            k_list.append(
                torch.tensor(fisheye_param.get_intrinsic(self.cam_k_dim)))
            r_list.append(torch.tensor(fisheye_param.get_extrinsic_r()))
            t_list.append(torch.tensor(fisheye_param.get_extrinsic_t()))
        k_tensor = torch.stack(k_list)
        r_tensor = torch.stack(r_list)
        t_tensor = torch.stack(t_list)
        # prepare kps3d
        keypoints3d = self.gt3d[scene_idx]
        kps3d = keypoints3d.get_keypoints()[frame_idx]
        # if this frame is the end of clip(scene)
        end_of_clip = end_of_clip and not self.shuffled
        # prepare keyword data
        kw_data = {}

        if self.kps2d_convention is not None:
            mview_keypoints2d_list = self.percep_keypoints2d[scene_idx]
            # mview_kps2d_list = []
            # for _, keypoints2d in enumerate(mview_keypoints2d_list):
            #     mview_kps2d_list.append(keypoints2d.get_keypoints()[frame_idx])
            mview_kps2d_list = []
            n_view = mview_img_tensor.shape[0]
            for view_idx in range(n_view):
                mview_kps2d_list.append(mview_keypoints2d_list[view_idx][frame_idx])
            kw_data = mview_kps2d_list
        # import pdb; pdb.set_trace()
        return mview_img_tensor, k_tensor, r_tensor,\
            t_tensor, kps3d, end_of_clip, kw_data

    def load_image_list(self):
        """Load multi-scene image lists."""
        mscene_list = []
        for scene_idx in range(self.n_scene):
            scene_dir = os.path.join(self.meta_path, f'scene_{scene_idx}')
            image_list_names = glob.glob(
                os.path.join(scene_dir, 'image_list_view_*'))
            n_view = len(image_list_names)
            # index of mview_list is view idx
            mview_list = []
            for idx in range(n_view):
                view_idx = self.view_idxs[idx]
                with open(
                        os.path.join(scene_dir,
                                     f'image_list_view_{view_idx:02d}.txt'),
                        'r') as f_read:
                    lines = f_read.readlines()
                mview_list.append(lines)
            # index of mframe_list is frame idx
            mframe_list = []
            for frame_idx in range(len(mview_list[0])):
                sframe_list = []
                for idx in range(n_view):
                    sframe_list.append(mview_list[idx][frame_idx].strip())
                mframe_list.append(sframe_list)
            # mframe_list = mframe_list[:5] #for debug 
            mscene_list.append(mframe_list)
        self.image_list = mscene_list

    def init_index_mapping(self):
        """Init a list mapping dataset index to scene index and frame index."""
        mscene_len = []
        total_len = 0
        for mframe_list in self.image_list:
            mscene_len.append(len(mframe_list))
            total_len += mscene_len[-1]
        self.index_mapping = mscene_len
        self.len = total_len

    def __len__(self) -> int:
        return self.len

    def process_index_mapping(self, index: int) -> Tuple[int, int, bool]:
        """Map dataset index to scene index and frame index.

        Args:
            index (int):
                Index in dataset. It must be lower than len(self).

        Raises:
            IndexError: index >= len(self)

        Returns:
            Tuple[int, int, bool]:
                Scene index.
                Frame index in the scene.
                Whether this moment is the last frame of the clip.
        """
        for scene_idx, scene_len in enumerate(self.index_mapping):
            if index < scene_len:
                return scene_idx, index, index == scene_len - 1
            else:
                index -= scene_len
        self.logger.error(f'Index out of range: {index}\n' +
                          f'len(dataset): {self.len}')
        raise IndexError

    def load_camera_parameters(self):
        """Load multi-scene fisheye parameters."""
        mscene_list = []
        for scene_idx in range(self.n_scene):
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
                            re.search(r'fisheye_param_([0-9]+).json',
                                      file_name).group(1)))
            n_view = len(cam_names)
            self.view_idxs = view_idxs
            mview_list = []
            for idx in range(n_view):
                view_idx = self.view_idxs[idx]
                fisheye_param = FisheyeCameraParameter.fromfile(
                    os.path.join(cam_dir,
                                 f'fisheye_param_{view_idx:02d}.json'))
                translation = self.factor * np.array(
                    fisheye_param.get_extrinsic_t())
                fisheye_param.set_KRT(T=translation)
                if fisheye_param.world2cam != self.cam_world2cam:
                    fisheye_param.inverse_extrinsic()
                mview_list.append(fisheye_param)
            mscene_list.append(mview_list)
        self.fisheye_params = mscene_list

    def load_ground_truth_3d(self):
        """Load multi-scene ground truth keypoints3d."""
        mscene_list = []
        for scene_idx in range(self.n_scene):
            keypoints3d_path = os.path.join(self.meta_path,
                                            f'scene_{scene_idx}',
                                            'keypoints3d_GT.npz')
            keypoints3d = Keypoints.fromfile(keypoints3d_path)
            keypoints3d.logger = self.logger
            keypoints3d = keypoints3d.to_tensor()
            if self.gt_kps3d_convention is not None:
                keypoints3d = convert_keypoints(
                    keypoints=keypoints3d,
                    dst=self.gt_kps3d_convention,
                    approximate=True)
            # save mask info into confidence
            kps3d_mask = keypoints3d.get_mask()
            kps3d = keypoints3d.get_keypoints()
            kps3d[..., -1] = kps3d[..., -1] * kps3d_mask
            kps3d[..., :3] = self.factor * kps3d[..., :3]
            keypoints3d.set_keypoints(kps3d)
            mscene_list.append(keypoints3d)
        self.gt3d = mscene_list

    def load_perception_2d(self):
        """Load multi-scene keypoints2d and paf."""

        mscene_keypoints_list = []
        for scene_idx in range(self.n_scene):

            pkl_names = glob.glob(
                os.path.join(self.meta_path, f'scene_{scene_idx}', "kps2d_paf", '*pkl'))
            n_view = len(pkl_names)
            mview_kps2d = []
            for view_idx in range(n_view):

                img_size = (self.fisheye_params[scene_idx][view_idx].width, self.fisheye_params[scene_idx][view_idx].height)

                file_name = os.path.join(self.meta_path, f'scene_{scene_idx}', "kps2d_paf", str(view_idx)+'.pkl')
                f = open(file_name,'rb')
                detections = pickle.load(f)
                
                frame_size = len(detections)

                joint_size = 25
                paf_size =26

                convert_joint_size = 18
                convert_paf_size = 19

                convert_detections = []
                for i in range(frame_size):
                    var = {'joints':{j:[] for j in range(convert_joint_size)}, 'pafs':{k:[] for k in range(convert_paf_size)}}
                    convert_detections.append(var)

                for frame_id in range(frame_size):
                    for joint_id in range(joint_size):
                        #resize
                        detections[frame_id]['joints'][joint_id][:,0] = detections[frame_id]['joints'][joint_id][:,0]*(img_size[0] - 1)
                        detections[frame_id]['joints'][joint_id][:,1] = detections[frame_id]['joints'][joint_id][:,1] * (img_size[1] - 1)

                    # for paf_id in range(paf_size):
                    #     detections[frame_id]['pafs'][paf_id] = np.power(detections[frame_id]['pafs'][paf_id], 0.8)

                    #convert keypoint type body25-skel19
                    joint_mapping = [4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, -1, -1, 18, -1, -1]
                    pafs_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, 16, -1, -1, 17, -1, -1]

                    for joint_id in range(joint_size):
                        tmp_joint_id = joint_mapping[joint_id]
                        if tmp_joint_id != -1:
                            convert_detections[frame_id]['joints'][tmp_joint_id] = detections[frame_id]['joints'][joint_id]
                    
                    for paf_id in range(paf_size):
                        tmp_paf_id = pafs_mapping[paf_id]
                        if tmp_paf_id != -1: 
                            convert_detections[frame_id]['pafs'][tmp_paf_id]= detections[frame_id]['pafs'][paf_id]
                f.close()
                mview_kps2d.append(convert_detections)
            mscene_keypoints_list.append(mview_kps2d)
        self.percep_keypoints2d = mscene_keypoints_list 
