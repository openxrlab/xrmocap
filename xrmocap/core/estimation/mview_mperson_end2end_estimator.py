# yapf: disable
import copy
import logging
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torchvision.transforms import Compose
from tqdm import tqdm
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.io.image import (
    get_n_frame_from_mview_src, load_clip_from_mview_src,
)
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.transform.image.builder import build_image_transform
from xrmocap.transform.image.shape import get_affine_trans_aug
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from xrmocap.utils.geometry import get_scale
from xrmocap.utils.mvp_utils import norm2absolute, process_dict
from .mperson_smpl_estimator import MultiPersonSMPLEstimator

# yapf: enable


class MultiViewMultiPersonEnd2EndEstimator(MultiPersonSMPLEstimator):
    """Api for estimating keypoints3d and smpl in a multi-view multi-person
    scene, using end2end learning-based method."""

    def __init__(
        self,
        work_dir: str,
        img_pipeline: dict,
        image_size: List[int],
        heatmap_size: List[int],
        n_max_person: int,
        n_kps: int,
        kps3d_convention: str,
        kps3d_model: Union[dict, torch.nn.Module],
        kps3d_model_path: Union[None, str],
        dataset: Union[None, str] = None,
        inference_conf_thr: float = 0.0,
        load_batch_size: int = 1,
        kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]], None] = None,
        smplify: Union[None, dict, SMPLify] = None,
        cam_metric_unit: str = 'millimeter',
        verbose: bool = True,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Initialization of the class.

        Args:
            work_dir (str):
                Path to the folder for running the api. No file in work_dir
                will be modified
                or added by MultiViewSinglePersonSMPLEstimator.
            img_pipeline (dict):
                Image pre-processin pipeline for input images,
                need to align with configs during model training.
            dataset (str):
                Name of the dataset.
            image_size (List[int]):
                Input image size, need to align with configs during
                    model training.
            heatmap_size (List[int]):
                Heatmap size, need to align with configs during model training.
            n_max_person (int):
                Number of maximum person the model can handle.
            n_kps (int):
                Number of keypoints.
            kps3d_convention (str):
                Convention of keypoints.
            kps3d_model (Union[dict, torch.nn.Module]):
                An end-to-end mview mperson keypoints3d predicting model.
            kps3d_model_path (Union[None, str]):
                Path to pre-trained model for inference.
            inference_conf_thr (float, optional):
                Threshold for confidence whether prediction is a human.
                Defaults to 0.0.
            load_batch_size (int, optional):
                Batch size in image loading. Defaults to 1.
            kps3d_optimizers (Union[List[Union[BaseOptimizer, dict]],
                None], optional):
                A list of keypoints3d optimizers or their configs. If given,
                keypoints3d will be
                optimized by the cascaded final optimizers before estimation.
                Defaults to None.
            smplify (Union[None, dict, SMPLify], optional):
                A SMPLify instance or its config. Defaults to None.
            cam_metric_unit (str, optional):
                Metric unit for camera parameters. Defaults to 'millimeter'.
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

        if torch.cuda.is_available():
            self.logger.info('CUDA is available')
        else:
            self.logger.warning('CUDA is not available, running on CPU')

        self.logger = get_logger(logger)

        self.dataset = dataset
        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.inference_conf_thr = inference_conf_thr
        self.kps3d_convention = kps3d_convention
        self.n_max_person = n_max_person
        self.n_kps = n_kps

        if cam_metric_unit == 'meter':
            self.factor = 1.0
        elif cam_metric_unit == 'centimeter':
            self.factor = 100.0
        elif cam_metric_unit == 'millimeter':
            self.factor = 1000.0

        # mvp model only accept batch size = 1 and views in the format of list
        self.load_batch_size = load_batch_size
        if self.load_batch_size != 1:
            self.logger.error('Please set load_batch_size to' +
                              '1 for end2end estimator')
            raise ValueError

        if isinstance(kps3d_model, dict):
            if kps3d_model_path is None:
                self.logger.error('Please define a pretrained model')
                raise ValueError
            kps3d_model_setup = dict(
                type='MviewPoseTransformer',
                is_train=False,
                logger=self.logger)
            kps3d_model_setup.update(kps3d_model)
            self.kps3d_model = build_architecture(kps3d_model_setup)
            self.logger.info(f'Load saved models state {kps3d_model_path}')
            load_checkpoint(
                self.kps3d_model,
                kps3d_model_path,
                logger=self.logger,
                map_location='cpu')
        else:
            self.kps3d_model = kps3d_model

        self.img_pipeline = []
        for transform in img_pipeline:
            if isinstance(transform, dict):
                transform = build_image_transform(transform)
            self.img_pipeline.append(transform)
        self.img_pipeline = Compose(self.img_pipeline)

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

        kps3d_batch = np.full((n_frame, self.n_max_person, self.n_kps, 4),
                              np.nan)

        # load from scene input
        for start_idx in tqdm(range(0, n_frame, self.load_batch_size)):
            end_idx = min(n_frame, start_idx + self.load_batch_size)
            mview_batch_arr = load_clip_from_mview_src(
                start_idx=start_idx,
                end_idx=end_idx,
                img_arr=img_arr,
                img_paths=img_paths,
                video_paths=video_paths,
                logger=self.logger)
            _, _, h, w, c = mview_batch_arr.shape

            # prepare input data, img list and meta list
            list_inputs = []
            for _, img in enumerate(mview_batch_arr.squeeze()):
                img_tensor = self.img_pipeline(img)
                img_tensor = img_tensor.unsqueeze(0)
                list_inputs.append(img_tensor)
            meta = self.prepare_meta(cam_param, h, w)

            frame_valid_pred_kps3d, _ = self.estimate_perception3d(
                list_inputs, meta, self.inference_conf_thr)

            pred_n_person = frame_valid_pred_kps3d.shape[0]
            kps3d_batch[start_idx, :pred_n_person,
                        ...] = frame_valid_pred_kps3d

        # Convert array to keypoints instance
        pred_keypoints3d = Keypoints(
            dtype='numpy',
            kps=kps3d_batch,
            mask=kps3d_batch[..., -1] > 0,
            convention=self.kps3d_convention,
            logger=self.logger)

        # Optimizing keypoints3d
        pred_keypoints3d = self.optimize_keypoints3d(pred_keypoints3d)

        # Fitting SMPL model
        smpl_data_list = self.estimate_smpl(keypoints3d=pred_keypoints3d)

        return pred_keypoints3d, smpl_data_list

    def estimate_perception3d(self,
                              img_arr: Union[None, np.ndarray],
                              meta: Union[None, dict],
                              threshold: float = 0.0):
        """Estimate perception3d from images per frame.

        Args:
            img_arr (Union[None, np.ndarray]):
                Input images.
            meta (Union[None, dict]):
                Input meta data.
            threshold (float, optional):
                Threshold for a valid person. Defaults to 0.0.

        Returns:
            Union[np.ndarray, torch.Tensor]:
                All predicted 3D keypoints per frame [n_person, n_kps, 4]
        """
        if torch.cuda.is_available():
            self.kps3d_model.cuda()
            img_arr = [i.to('cuda') for i in img_arr]
            meta = [
                process_dict(t, device='cuda', dummy_dim=None, dtype=None)
                for t in meta
            ]
        self.kps3d_model.eval()

        output = self.kps3d_model(views=img_arr, meta=meta)

        bs, n_queries = output['pred_logits'].shape[:2]
        src_poses = output['pred_poses']['outputs_coord']. \
            view(bs, n_queries, self.n_kps, 3)
        src_poses = norm2absolute(src_poses, self.kps3d_model.grid_size,
                                  self.kps3d_model.grid_center)

        if 'panoptic' in self.dataset:
            trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                         [0.0, -1.0, 0.0]]).double()
            src_poses = torch.mm(src_poses, trans_ground)

        score = output['pred_logits'][:, :, 1:2].sigmoid()
        score = score.unsqueeze(2).expand(-1, -1, self.n_kps, -1)

        pred_kps3d = torch.cat([src_poses, score - threshold], dim=-1)
        pred_kps3d = pred_kps3d.detach().cpu().numpy()

        frame_pred_kps3d = pred_kps3d[0]
        frame_valid_pred_kps3d = frame_pred_kps3d[frame_pred_kps3d[:, 0,
                                                                   3] >= 0]

        return frame_valid_pred_kps3d / self.factor, frame_pred_kps3d

    def prepare_meta(self, cam_param: List[FisheyeCameraParameter],
                     height: int, width: int):
        """Prepare meta data for end2end estimator.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                Camera parameters.
            height (int):
                Image height.
            width (int):
                Image width.
        """
        meta = []
        for cam_idx, input_camera in enumerate(cam_param):

            camera = input_camera.clone()
            if not camera.world2cam:
                camera.inverse_extrinsic()

            kw_data = {}
            k_tensor = torch.tensor(camera.get_intrinsic(k_dim=3))
            r_tensor = torch.tensor(camera.get_extrinsic_r())
            t_tensor = self.factor * torch.tensor(camera.get_extrinsic_t())

            k_tensor = k_tensor.double()
            r_tensor = r_tensor.double()
            t_tensor = t_tensor.double()

            dist_coeff_tensor = torch.tensor(camera.get_dist_coeff())

            c = np.array([width / 2.0, height / 2.0])
            s = get_scale(np.array([width, height]), self.image_size)
            r = 0  # NOTE: do not apply rotation augmentation
            hm_scale = self.heatmap_size / self.image_size

            # Affine transformations
            trans, inv_trans, aug_trans = get_affine_trans_aug(
                c, s, hm_scale, self.image_size, r)
            kw_data['affine_trans'] = trans
            kw_data['inv_affine_trans'] = inv_trans
            kw_data['aug_trans'] = aug_trans
            kw_data['center'] = c
            kw_data['scale'] = s

            view_kw_data = copy.deepcopy(kw_data)

            # Camera parameters
            view_kw_data['camera'] = dict()
            if 'panoptic' in self.dataset:
                trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                                             [0.0, 1.0, 0.0]]).double()
            elif 'shelf' or 'campus' in self.dataset:
                trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]]).double()
            else:
                self.logger.warning(
                    'This dataset is not supported by default, '
                    'please set the ground coordinate transformation. '
                    'Using Eye matrix by default.')
                trans_ground = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]]).double()

            r_trans = torch.mm(r_tensor, trans_ground)
            t_trans = -torch.mm(r_trans.T, t_tensor.reshape((3, 1)))

            view_kw_data['camera']['camera_standard_T'] = t_tensor  # world2cam
            view_kw_data['camera']['R'] = r_trans
            view_kw_data['camera']['T'] = t_trans  # cam2world
            view_kw_data['camera']['K'] = k_tensor
            view_kw_data['camera']['dist_coeff'] = dist_coeff_tensor

            meta.append(view_kw_data)

        meta = [
            process_dict(t, device=None, dummy_dim=0, dtype='tensor')
            for t in meta
        ]

        return meta
