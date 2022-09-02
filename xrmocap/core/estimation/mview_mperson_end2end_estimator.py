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
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .base_estimator import BaseEstimator

# yapf: enable


class MultiViewMultiPersonEnd2EndEstimator(BaseEstimator):

    def __init__(self,
                 work_dir: str,
                 smplify: Union[dict, SMPLify],
                 kps3d_model: Union[dict, torch.nn.module],
                 load_batch_size: int = 500,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Api for estimating smpl parameters in a multi-view, single-person
        scene.

        Args:
            work_dir (str):
                Path to the folder for running the api. No file in work_dir
                will be modified
                or added by MultiViewSinglePersonSMPLEstimator.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            kps3d_model (Union[dict, torch.nn.module]):
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
        super().__init__(work_dir, verbose, logger)

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
        """Run mutli-view single-person smpl estimator once. run() needs one
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

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **optim_kwargs) -> Keypoints:
        """Optimize keypoints3d.

        Args:
            keypoints3d (Keypoints): A keypoints3d Keypoints instance
        Returns:
            Keypoints: The optimized keypoints3d.
        """
        if self.kps3d_optimizers is not None:
            for optimizer in self.kps3d_optimizers:
                keypoints3d = optimizer.optimize_keypoints3d(
                    keypoints3d, **optim_kwargs)
        return keypoints3d

    def estimate_smpl(self, keypoints3d: Keypoints) -> SMPLData:
        """Estimate smpl parameters according to keypoints3d.

        Args:
            keypoints3d (Keypoints):
                A keypoints3d Keypoints instance, with only one person
                inside. This method will take the person at
                keypoints3d.get_keypoints()[:, 0, ...] to run smplify.

        Returns:
            SMPLData:
                SMPL data of the person.
        """
        self.logger.info('Estimating SMPL.')
        working_convention = 'smpl'

        n_frame = keypoints3d.get_frame_number()
        n_person = keypoints3d.get_person_number()
        keypoints3d = keypoints3d.to_tensor(device=self.smplify.device)
        person_mask = keypoints3d.get_mask()
        person_mask = torch.sum(person_mask, dim=2) > 0

        keypoints3d = convert_keypoints(
            keypoints=keypoints3d, dst=working_convention)
        kps3d_tensor = keypoints3d.get_keypoints()[:, :, :, :3].float()
        kps3d_conf = keypoints3d.get_mask()[:, :, ...]

        smpl_data_list = []
        for person in range(n_person):
            if person_mask[:, person].sum() == 0:
                continue
            global_orient = torch.zeros((n_frame, 3)).to(self.smplify.device)
            transl = torch.full((n_frame, 3), 1000.0).to(self.smplify.device)
            body_pose = torch.zeros((n_frame, 69)).to(self.smplify.device)
            betas = torch.zeros((n_frame, 10)).to(self.smplify.device)
            s_kps3d_tensor = kps3d_tensor[:, person][person_mask[:, person]]
            s_kps3d_conf = kps3d_conf[:, person][person_mask[:, person]]
            # build and run
            kp3d_mse_input = build_handler(
                dict(
                    type='Keypoint3dMSEInput',
                    keypoints3d=s_kps3d_tensor,
                    keypoints3d_conf=s_kps3d_conf,
                    keypoints3d_convention=working_convention,
                    handler_key='keypoints3d_mse'))
            kp3d_llen_input = build_handler(
                dict(
                    type='Keypoint3dLimbLenInput',
                    keypoints3d=s_kps3d_tensor,
                    keypoints3d_conf=s_kps3d_conf,
                    keypoints3d_convention=working_convention,
                    handler_key='keypoints3d_limb_len'))
            registrant_output = self.smplify(
                input_list=[kp3d_mse_input, kp3d_llen_input])

            smpl_data = SMPLData()
            global_orient[
                person_mask[:, person]] = registrant_output['global_orient']
            transl[person_mask[:, person]] = registrant_output['transl']
            body_pose[person_mask[:, person]] = registrant_output['body_pose']
            betas[person_mask[:, person]] = registrant_output['betas']
            output = {}
            output['global_orient'] = global_orient
            output['transl'] = transl
            output['body_pose'] = body_pose
            output['betas'] = betas
            smpl_data.from_param_dict(output)
            smpl_data_list.append(smpl_data)
        return smpl_data_list
