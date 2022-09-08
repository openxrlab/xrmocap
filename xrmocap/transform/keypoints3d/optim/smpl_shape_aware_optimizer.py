# yapf: disable
import logging
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.model.body_model.builder import SMPL, build_body_model
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.ops.projection.builder import PytorchProjector, build_projector
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from .base_optimizer import BaseOptimizer


# yapf: enable
class SMPLShapeAwareOptimizer(BaseOptimizer):

    def __init__(self,
                 smplify: Union[dict, SMPLify],
                 body_model: Union[dict, SMPL],
                 projector: Union[dict, PytorchProjector],
                 iteration=1,
                 refine_threshold=1,
                 kps2d_conf_threshold=0.97,
                 use_percep2d_optimizer: bool = False,
                 steps: int = 1000,
                 lr: float = 1e-3,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """SMPL shape-aware optimizer.

        Args:
            smplify (Union[dict, SMPLify]): A SMPLify instance or its config.
            body_model (Union[dict, SMPL]):
                An instance of SMPL body_model or a config dict
                of SMPL body_model.
            projector (Union[dict, PytorchProjector]):
                An instance of PytorchProjector projector for points projection
                or a config dict of PytorchProjector projector.
            iteration (int, optional):
                The number of iterations optimized. Defaults to 1.
            refine_threshold (int, optional): Defaults to 1.
            kps2d_conf_threshold (float, optional):
                Threshold for 3D optimization using SMPL model.
                Defaults to 0.97.
            use_percep2d_optimizer (bool, optional):
                Whether to use 2D perception data for optimization.
                Defaults to False.
            steps (int, optional): Defaults to 1000.
            lr (float, optional): step size for Adam optimizer.
                Defaults to 1e-3.
            verbose (bool, optional): Whether to log info. Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        if isinstance(smplify, dict):
            smplify['logger'] = logger
            self.smplify = build_registrant(smplify)
        else:
            self.smplify = smplify
        if isinstance(body_model, dict):
            body_model['logger'] = logger
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = body_model

        if isinstance(projector, dict):
            self.projector = build_projector(projector)
        else:
            self.projector = projector
        self.device = self.smplify.device
        self.refine_threshold = refine_threshold
        self.kps2d_conf_threshold = kps2d_conf_threshold
        self.iteration = iteration
        self.use_percep2d_optimizer = use_percep2d_optimizer
        self.steps = steps
        self.lr = lr

    def prepare_data(
        self, skps3d: np.ndarray, frame_idx: int, kp3d_idx: int,
        sperson: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], np.ndarray, list]:
        """Prepare data for one keypoint.

        Args:
            skps3d (np.ndarray): The keypoints3d from single person,
                in shape (n_kps3d, 3).
            frame_idx (int): The frame index.
            kp3d_idx (int): The keypoints 3d index.
            sperson (List[np.ndarray]): The matched person id.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], np.ndarray, list]:
                A keypoint3d,  keypoints2d from different views, the
                selected views and keypoints2d conf from different views.
        """

        mview_n_person = len(np.where(~np.isnan(sperson))[0])
        reprojected_error = np.zeros(mview_n_person)
        mview_person_idx = 0
        kps2d_tensor = []
        mkps2d_conf_list = []
        for view, kps2d_idx in enumerate(sperson):
            if np.isnan(kps2d_idx):
                kps2d_tensor.append(torch.tensor(np.nan))
                continue
            f_person_idx = self.mview_person_id[view][frame_idx][int(
                sperson[view])]
            kp2d = self.kps2d_list[view][frame_idx, f_person_idx, kp3d_idx]
            mkps2d_conf_list.append(self.kps2d_conf_list[view][frame_idx,
                                                               f_person_idx,
                                                               kp3d_idx])

            kps2d_tensor.append(torch.tensor(kp2d, dtype=torch.float32))
            proj_kp2d = self.projector.project_single_point(
                torch.from_numpy(skps3d[kp3d_idx])).numpy()

            reprojected_error[mview_person_idx] += np.linalg.norm(
                proj_kp2d[view] - kp2d)
            mview_person_idx += 1
        selected_view = (reprojected_error - reprojected_error.mean()
                         ) / reprojected_error.std() < self.refine_threshold
        selected_view = selected_view & (
            reprojected_error < reprojected_error.mean())

        kp3d_tensor = torch.tensor(skps3d[kp3d_idx], dtype=torch.float32)
        return kp3d_tensor, kps2d_tensor, selected_view, mkps2d_conf_list

    def step(self, keypoints3d: Keypoints, keypoints2d: List[Keypoints],
             matched_list, keypoints_from_smpl: Keypoints) -> Keypoints:

        kps3d = keypoints3d.get_keypoints()[..., :3]
        kps3d_conf = keypoints3d.get_keypoints()[..., 3:4]
        kps3d_mask = keypoints3d.get_mask()
        n_kps3d = keypoints3d.get_keypoints_number()
        convention = keypoints3d.get_convention()

        kps3d_from_smpl = keypoints_from_smpl.get_keypoints()[..., :3]
        self.kps2d_list = []
        self.kps2d_conf_list = []
        for keypoints in keypoints2d:
            kps2d = keypoints.get_keypoints()[..., :2]
            kps2d_conf = keypoints.get_keypoints()[..., 2]
            self.kps2d_list.append(kps2d)
            self.kps2d_conf_list.append(kps2d_conf)
        for frame_idx, person in enumerate(tqdm(matched_list)):
            f_person_mask = np.where(
                np.sum(kps3d_mask[frame_idx], axis=-1) > 0)[0]
            for person_idx, sperson in enumerate(person):
                skps3d = kps3d[frame_idx, f_person_mask[person_idx]]
                for kp3d_idx in range(n_kps3d):
                    kp3d_tensor, kps2d_tensor, selected_view, mkps2d_conf_list\
                        = self.prepare_data(skps3d, frame_idx,
                                            kp3d_idx, sperson)

                    if self.use_percep2d_optimizer and selected_view.sum() > 0:
                        refined_kp3d = self.optimizer_based_percep2d(
                            kp3d_tensor, kps2d_tensor, selected_view)
                        kps3d[frame_idx, f_person_mask[person_idx],
                              kp3d_idx] = refined_kp3d.numpy().reshape(-1)
                    else:
                        high_level_conf = np.where(
                            np.array(mkps2d_conf_list) >
                            self.kps2d_conf_threshold)[0]
                        if len(high_level_conf) < 2:
                            kp3d_tensor_from_smpl = torch.tensor(
                                kps3d_from_smpl[frame_idx,
                                                f_person_mask[person_idx],
                                                kp3d_idx],
                                dtype=torch.float32)
                            refined_kp3d = self.optimizer_based_smpl(
                                kp3d_tensor, kp3d_tensor_from_smpl)
                            kps3d[frame_idx, f_person_mask[person_idx],
                                  kp3d_idx] = refined_kp3d.numpy().reshape(-1)

        kps3d = np.concatenate((kps3d, kps3d_conf), axis=-1)
        ret_keypoints = Keypoints(
            kps=kps3d, mask=kps3d_mask, convention=convention)
        return ret_keypoints

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             keypoints2d: List[Keypoints],
                             mview_person_id: list, matched_list: list,
                             cam_params: List[FisheyeCameraParameter],
                             **kwargs: dict) -> Keypoints:
        """Forward function of keypoints3d optimizer.

        Args:
            keypoints3d (Keypoints): A keypoints3d Keypoints instance.
            keypoints2d (Keypoints): A list of keypoints2d Keypoints instance.
            mview_person_id (list):
                The kps2d index in the bbox2d threshold of different frames
                from different views.
            matched_list (list): The matched kps2d index from different views,
                np.nan indicates that no person is observed in this view.
            cam_params (List[FisheyeCameraParameter]): Camera parameter.
            kwargs:
                Redundant keyword arguments to be
                ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        self.mview_person_id = mview_person_id
        self.projector.set_cameras(cam_params)
        for step in range(self.iteration):
            smpl_data_list = self.estimate_smpl(keypoints3d=keypoints3d)
            kps3d_list = []
            for smpl_data in smpl_data_list:
                body_model_kwargs = smpl_data.to_tensor_dict()
                body_model_output = self.body_model(**body_model_kwargs)
                model_joints = body_model_output['joints']
                kps3d_list.append(model_joints)
            kps3d = torch.stack(kps3d_list, dim=1).detach().numpy()
            kps3d_ones = np.ones_like(kps3d[..., 0:1])
            kps3d = np.concatenate((kps3d, kps3d_ones), axis=-1)
            keypoints3d_45 = Keypoints(kps=kps3d, convention='smpl_45')
            smpl_keypoints3d = convert_keypoints(
                keypoints3d_45, 'coco', approximate=True)
            kps3d_conf = np.ones_like(smpl_keypoints3d.get_keypoints()[...,
                                                                       0:1])
            smpl_keypoints3d.set_keypoints(
                np.concatenate(
                    (smpl_keypoints3d.get_keypoints()[..., :3], kps3d_conf),
                    axis=-1))

            keypoints3d = self.step(keypoints3d, keypoints2d, matched_list,
                                    smpl_keypoints3d)
        return keypoints3d

    def optimizer_based_percep2d(self, init_kps3d, mview_kps2d_tensor,
                                 selected_view) -> torch.Tensor:

        optim_kps3d = nn.Parameter(init_kps3d)
        optimizer = optim.Adam([optim_kps3d], lr=self.lr)
        last_loss = torch.tensor(0.)

        for step in range(self.steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.)
            proj_kps2d = self.projector.project_single_point(optim_kps3d)
            for i, (is_selected, kps2d) in enumerate(
                    zip(selected_view, mview_kps2d_tensor)):
                if is_selected and not torch.isnan(kps2d).any():
                    loss += torch.norm(proj_kps2d[i] - kps2d)

            if torch.abs(last_loss - loss) < 1e-3:
                break
            else:
                last_loss = loss
            loss.backward()
            optimizer.step()
        return optim_kps3d.detach()

    def optimizer_based_smpl(self, init_kps3d,
                             kps3d_from_smpl) -> torch.Tensor:

        optim_kps3d = nn.Parameter(init_kps3d)
        optimizer = optim.Adam([optim_kps3d], lr=self.lr)
        last_loss = torch.tensor(0.)

        for step in range(self.steps):
            optimizer.zero_grad()
            loss = torch.linalg.norm(kps3d_from_smpl - optim_kps3d, ord=2)

            if torch.abs(last_loss - loss) < 1e-3:
                break
            else:
                last_loss = loss
            loss.backward()
            optimizer.step()
        return optim_kps3d.detach()

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
        keypoints3d = keypoints3d.to_tensor(device=self.device)
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
            global_orient = torch.zeros((n_frame, 3)).to(self.device)
            transl = torch.full((n_frame, 3), 1000.0).to(self.device)
            body_pose = torch.zeros((n_frame, 69)).to(self.device)
            betas = torch.zeros((n_frame, 10)).to(self.device)
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
