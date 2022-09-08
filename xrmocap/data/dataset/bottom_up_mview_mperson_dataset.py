import glob
import logging
import numpy as np
import os
import torch
import pickle
from typing import Tuple, Union
from .mview_mperson_dataset import MviewMpersonDataset

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class BottomUpMviewMpersonDataset(MviewMpersonDataset):

    def __init__(self,
                 data_root: str,
                 img_pipeline: list,
                 meta_path: str = 'xrmocap_meta',
                 test_mode: bool = True,
                 shuffled: bool = False,
                 metric_unit: Literal['meter', 'centimeter',
                                      'millimeter'] = 'meter',
                 kps2d_convention = 'fourd19',
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

        MviewMpersonDataset.__init__(
            self,
            data_root=data_root,
            img_pipeline=img_pipeline,
            meta_path=meta_path,
            test_mode=test_mode,
            shuffled=shuffled,
            metric_unit=metric_unit,
            kps2d_convention=kps2d_convention,
            gt_kps3d_convention=gt_kps3d_convention,
            cam_world2cam=cam_world2cam,
            cam_k_dim=cam_k_dim,
            logger=logger)
        
        
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

        mview_keypoints2d_list = self.percep_keypoints2d[scene_idx]
        mview_kps2d_list = []
        n_view = mview_img_tensor.shape[0]
        for view_idx in range(n_view):
            mview_kps2d_list.append(mview_keypoints2d_list[view_idx][frame_idx])
        kw_data = mview_kps2d_list
        
        return mview_img_tensor, k_tensor, r_tensor,\
            t_tensor, kps3d, end_of_clip, kw_data

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
                n_kps = 25
                n_pafs =26
                convert_n_kps = 18
                convert_n_pafs = 19
                convert_detections = []
                for i in range(frame_size):
                    var = {'joints':{j:[] for j in range(convert_n_kps)}, 'pafs':{k:[] for k in range(convert_n_pafs)}}
                    convert_detections.append(var)

                for frame_id in range(frame_size):
                    for joint_id in range(n_kps):
                        #resize
                        detections[frame_id]['joints'][joint_id][:,0] = detections[frame_id]['joints'][joint_id][:,0]*(img_size[0] - 1)
                        detections[frame_id]['joints'][joint_id][:,1] = detections[frame_id]['joints'][joint_id][:,1] * (img_size[1] - 1)

                    #convert keypoint type body25-skel19
                    joint_mapping = [4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, -1, -1, 18, -1, -1]
                    pafs_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, 16, -1, -1, 17, -1, -1]

                    for joint_id in range(n_kps):
                        tmp_joint_id = joint_mapping[joint_id]
                        if tmp_joint_id != -1:
                            convert_detections[frame_id]['joints'][tmp_joint_id] = detections[frame_id]['joints'][joint_id]
                    
                    for paf_id in range(n_pafs):
                        tmp_paf_id = pafs_mapping[paf_id]
                        if tmp_paf_id != -1: 
                            convert_detections[frame_id]['pafs'][tmp_paf_id]= detections[frame_id]['pafs'][paf_id]
                f.close()
                mview_kps2d.append(convert_detections)
            mscene_keypoints_list.append(mview_kps2d)
        self.percep_keypoints2d = mscene_keypoints_list 
