# yapf: disable
import logging
import torch
from typing import List, Union

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .base_estimator import BaseEstimator

# yapf: enable


class MultiPersonSMPLEstimator(BaseEstimator):
    """Api for estimating smpl in a multi-person scene."""

    def __init__(self,
                 work_dir: str,
                 smplify: Union[dict, SMPLify],
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
