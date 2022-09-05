# yapf: disable
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.ffmpeg_utils import video_to_array

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.human_perception.builder import (
    MMdetDetector, MMposeTopDownEstimator, build_detector,
)
from xrmocap.io.image import load_multiview_images
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.ops.top_down_association.builder import (
    MvposeAssociator, build_top_down_associator,
)
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_num,
)
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .base_estimator import BaseEstimator

# yapf: enable


class MMSE(BaseEstimator):

    def __init__(self,
                 bbox_thr: float,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 associator: Union[dict, MvposeAssociator],
                 smplify: Union[dict, SMPLify],
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 pred_kps3d_convention: str = 'coco',
                 load_batch_size: int = 500,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:

        super().__init__(work_dir, verbose, logger)
        self.bbox_thr = bbox_thr
        self.load_batch_size = load_batch_size
        self.pred_kps3d_convention = pred_kps3d_convention

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

        if isinstance(associator, dict):
            associator['logger'] = logger
            self.associator = build_top_down_associator(associator)
        else:
            self.associator = associator

        if isinstance(smplify, dict):
            smplify['logger'] = logger
            self.smplify = build_registrant(smplify)
        else:
            self.smplify = smplify

        if kps3d_optimizers is None:
            self.kps3d_optimizers = None
        else:
            self.kps3d_optimizers = []
            for kps3d_optim in kps3d_optimizers:
                if isinstance(kps3d_optim, dict):
                    kps3d_optim['logger'] = logger
                    kps3d_optim = build_keypoints3d_optimizer(kps3d_optim)
                self.kps3d_optimizers.append(kps3d_optim)

    @overload
    def run(
        self, img_arr: np.ndarray, cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        ...

    @overload
    def run(
        self, img_paths: List[List[str]],
        cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        ...

    @overload
    def run(
        self, video_path: List[str], cam_param: List[FisheyeCameraParameter]
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
        ...

    def run(
        self,
        cam_param: List[FisheyeCameraParameter],
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[List[str]]] = None,
        video_paths: Union[None, List[str]] = None,
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:

        input_list = [img_arr, img_paths, video_paths]
        input_count = 0
        for input_instance in input_list:
            if input_instance is not None:
                input_count += 1
        if input_count > 1:
            self.logger.error('Redundant input!\n' +
                              'Please offer only one among' +
                              ' img_arr, img_paths and video_paths.')
            raise ValueError
        if img_arr is not None:
            mview_img_arr = img_arr
        elif img_paths is not None:
            mview_img_arr = load_multiview_images(img_paths)
        elif video_paths is not None:
            mview_img_list = []
            for video_path in video_paths:
                sv_img_arr = video_to_array(input_path=video_path)
                mview_img_list.append(sv_img_arr)
            mview_img_arr = np.asarray(mview_img_list)
        else:
            self.logger.error('No image input has been found!\n' +
                              'img_arr, img_paths and video_paths are None.')
            raise ValueError
        self.associator.set_cameras(cam_param)
        n_kps = get_keypoint_num(convention=self.pred_kps3d_convention)
        n_frame = mview_img_arr.shape[1]
        pred_kps3d = np.zeros(shape=(n_frame, 1, n_kps, 4))
        max_identity = 0
        for frame_idx in tqdm(range(n_frame)):
            mview_keypoints2d_list, mview_bbox2d_list = \
                self.estimate_keypoints2d(
                    img_arr=mview_img_arr[:, frame_idx:frame_idx + 1])
            keypoints2d_list = []
            bbox2d_list = []
            for view, keypoints2d in enumerate(mview_keypoints2d_list):
                if keypoints2d.get_convention() != self.pred_kps3d_convention:
                    keypoints2d = convert_keypoints(
                        keypoints=keypoints2d,
                        dst=self.pred_kps3d_convention,
                        approximate=True)
                keypoints2d_list.append(keypoints2d)
                bbox2d_list.append(torch.tensor(mview_bbox2d_list[view][0]))

            keypoints2d_idx, predict_keypoints3d, identities = \
                self.associator.associate_frame(
                    mview_img_arr=mview_img_arr[:, frame_idx].transpose(
                        0, 3, 1, 2),
                    mview_bbox2d=bbox2d_list,
                    mview_keypoints2d=keypoints2d_list,
                    affinity_type='geometry_mean'
                )
            for idx, identity in enumerate(identities):
                if identity > max_identity:
                    n_identity = identity - max_identity
                    pred_kps3d = np.concatenate(
                        (pred_kps3d,
                         np.zeros(shape=(n_frame, n_identity, n_kps, 4))),
                        axis=1)
                    max_identity = identity
                pred_kps3d[frame_idx,
                           identity] = predict_keypoints3d.get_keypoints()[0,
                                                                           idx]
        pred_keypoints3d = Keypoints(
            dtype='numpy',
            kps=pred_kps3d,
            mask=pred_kps3d[..., -1] > 0,
            convention=self.pred_kps3d_convention,
            logger=self.logger)
        return pred_keypoints3d

    def estimate_keypoints2d(
        self,
        img_arr: Union[None, np.ndarray] = None,
    ) -> List[Keypoints]:
        """Estimate keypoints2d in a top-down way.

        Args:
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
        Returns:
            List[Keypoints]:
                A list of keypoints2d instances.
        """
        mview_keypoints2d_list = []
        mview_bbox2d_list = []
        for view_idx in range(img_arr.shape[0]):
            view_img_arr = img_arr[view_idx]
            bbox_list = self.bbox_detector.infer_array(
                image_array=view_img_arr,
                disable_tqdm=(not self.verbose),
                multi_person=True,
            )
            kps2d_list, _, bbox2d_list = self.kps2d_estimator.infer_array(
                image_array=view_img_arr,
                bbox_list=bbox_list,
                disable_tqdm=(not self.verbose),
                return_heatmap=False,
            )
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            mview_keypoints2d_list.append(keypoints2d)
            mview_bbox2d_list.append(bbox2d_list)
        return mview_keypoints2d_list, mview_bbox2d_list
