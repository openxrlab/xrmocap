# yapf: disable
import logging
import numpy as np
import torch
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.io.image import (
    get_n_frame_from_mview_src, load_clip_from_mview_src,
)
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .mperson_smpl_estimator import MultiPersonSMPLEstimator

# yapf: enable


class MultiViewMultiPersonEnd2EndEstimator(MultiPersonSMPLEstimator):
    """Api for estimating keypoints3d and smpl in a multi-view multi-person
    scene, using end2end learning-based method."""

    def __init__(self,
                 work_dir: str,
                 smplify: Union[dict, SMPLify],
                 kps3d_model: Union[dict, torch.nn.Module],
                 load_batch_size: int = 500,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization of the class.

        Args:
            work_dir (str):
                Path to the folder for running the api. No file in work_dir
                will be modified
                or added by MultiViewSinglePersonSMPLEstimator.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            kps3d_model (Union[dict, torch.nn.Module]):
                An end-to-end mview mperson keypoints3d predicting model.
            kps3d_optimizers (Union[
                    List[Union[BaseOptimizer, dict]], None], optional):
                A list of keypoints3d optimizers or their configs. If given,
                keypoints3d will be
                optimized by the cascaded final optimizers before estimation.
                Defaults to None.
            verbose (bool, optional):
                Whether to print(logger.info) information during estimating.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(
            work_dir=work_dir,
            smplify=smplify,
            kps3d_optimizers=kps3d_optimizers,
            verbose=verbose,
            logger=logger)

        self.load_batch_size = load_batch_size
        if isinstance(kps3d_model, dict):
            self.kps3d_model = build_architecture(kps3d_model)
        else:
            self.kps3d_model = kps3d_model

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
    ) -> Tuple[List[Keypoints], List[SMPLData]]:
        """Run mutli-view multi-person end2end estimator once. run() needs one
        images input among [img_arr, img_paths, video_paths].

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
            img_paths (Union[None, List[List[str]]], optional):
                A nested list of image paths, in shape
                [n_view, n_frame]. Defaults to None.
            video_paths (Union[None, List[str]], optional):
                A list of video paths, each is a view.
                Defaults to None.

        Returns:
            Tuple[List[Keypoints], List[SMPLData]]:
                A list of keypoints3d, a list of SMPLData.
        """
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
        elif input_count < 1:
            self.logger.error('No image input has been found!\n' +
                              'img_arr, img_paths and video_paths are None.')
            raise ValueError
        n_frame = get_n_frame_from_mview_src(img_arr, img_paths, video_paths,
                                             self.logger)
        for start_idx in range(0, n_frame, self.load_batch_size):
            end_idx = min(n_frame, start_idx + self.load_batch_size)
            mview_batch_arr = load_clip_from_mview_src(
                start_idx=start_idx,
                end_idx=end_idx,
                img_arr=img_arr,
                img_paths=img_paths,
                video_paths=video_paths,
                logger=self.logger)
            n_view, n_frame, h, w, c = mview_batch_arr.shape
            # TODO: infer keypoints3d here
            kps3d_batch = None
        keypoints3d_list = [
            Keypoints(kps=kps3d_batch, mask=np.ones_like(kps3d_batch[..., 0]))
        ]
        smpl_data_list = [
            None,
        ] * len(keypoints3d_list)
        for person_idx, keypoints3d in enumerate(keypoints3d_list):
            keypoints3d = self.optimize_keypoints3d(keypoints3d)
            smpl_data = self.estimate_smpl(keypoints3d)
            keypoints3d_list[person_idx] = keypoints3d
            smpl_data_list[person_idx] = smpl_data
        return keypoints3d_list, smpl_data_list
