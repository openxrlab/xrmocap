# yapf: disable
import logging
import numpy as np
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.ffmpeg_utils import video_to_array

from xrmocap.data_structure.body_model import SMPLData, SMPLXData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.human_perception.builder import (
    MMdetDetector, MMposeTopDownEstimator, build_detector,
)
from xrmocap.io.image import load_multiview_images
from xrmocap.model.registrant.builder import SMPLify, build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, CameraErrorSelector, build_point_selector,
)
from xrmocap.transform.convention.keypoints_convention import convert_keypoints
from xrmocap.transform.keypoints3d.optim.builder import (
    BaseOptimizer, build_keypoints3d_optimizer,
)
from .base_estimator import BaseEstimator

# yapf: enable


class MultiViewSinglePersonSMPLEstimator(BaseEstimator):
    """Api for estimating smpl in a multi-view single-person scene."""

    def __init__(self,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 triangulator: Union[dict, BaseTriangulator],
                 smplify: Union[dict, SMPLify],
                 cam_pre_selector: Union[dict, BaseSelector, None] = None,
                 cam_selector: Union[dict, CameraErrorSelector, None] = None,
                 final_selectors: List[Union[dict, BaseSelector, None]] = None,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 load_batch_size: int = 500,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization of the class.

        Args:
            work_dir (str):
                Path to the folder for running the api. No file in work_dir
                will be modified
                or added by MultiViewSinglePersonSMPLEstimator.
            bbox_detector (Union[dict, MMdetDetector]):
                A human bbox_detector or its config.
            kps2d_estimator (Union[dict, MMposeTopDownEstimator]):
                A top-down kps2d estimator or its config.
            triangulator (Union[dict, BaseTriangulator]):
                A triangulator or its config.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            cam_pre_selector (Union[dict, BaseSelector, None], optional):
                A selector before selecting cameras. If it's given,
                points for camera selection will be filtered.
                Defaults to None.
            cam_selector (Union[dict, CameraErrorSelector, None], optional):
                A camera selector or its config. If it's given, cameras
                will be selected before triangulation.
                Defaults to None.
            final_selectors (List[Union[dict, BaseSelector, None]], optional):
                A list of selectors or their configs. If given, kps2d will be
                selected by the cascaded final selectors before triangulation.
                Defaults to None.
            kps3d_optimizers (Union[
                    List[Union[BaseOptimizer, dict]], None], optional):
                A list of keypoints3d optimizers or their configs. If given,
                keypoints3d will be
                optimized by the cascaded final optimizers after triangulation.
                Defaults to None.
            load_batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 500.
            verbose (bool, optional):
                Whether to print(logger.info) information during estimating.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(work_dir, verbose, logger)
        self.load_batch_size = load_batch_size

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

        if isinstance(triangulator, dict):
            triangulator['logger'] = logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        if isinstance(smplify, dict):
            smplify['logger'] = logger
            if smplify['type'].lower() == 'smplify':
                self.smpl_data_type = 'smpl'
            elif smplify['type'].lower() == 'smplifyx':
                self.smpl_data_type = 'smplx'
            else:
                self.logger.error('SMPL data type not found.')
                raise TypeError

            self.smplify = build_registrant(smplify)
        else:
            self.smplify = smplify

        if isinstance(cam_pre_selector, dict):
            cam_pre_selector['logger'] = logger
            self.cam_pre_selector = build_point_selector(cam_pre_selector)
        else:
            self.cam_pre_selector = cam_pre_selector

        if isinstance(cam_selector, dict):
            cam_selector['logger'] = logger
            cam_selector['triangulator']['camera_parameters'] = \
                self.triangulator.camera_parameters
            self.cam_selector = build_point_selector(cam_selector)
        else:
            self.cam_selector = cam_selector

        if final_selectors is None:
            self.final_selectors = None
        else:
            self.final_selectors = []
            for selector in final_selectors:
                if isinstance(selector, dict):
                    selector['logger'] = logger
                    selector = build_point_selector(selector)
                self.final_selectors.append(selector)

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
        init_smpl_data: Union[None, SMPLData] = None,
    ) -> Tuple[List[Keypoints], Keypoints, SMPLData]:
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
            Tuple[List[Keypoints], Keypoints, SMPLData]:
                A list of kps2d, an instance of Keypoints 3d,
                an instance of SMPLData.
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
        keypoints2d_list = self.estimate_keypoints2d(img_arr=mview_img_arr)
        keypoints3d = self.estimate_keypoints3d(
            cam_param=cam_param, keypoints2d_list=keypoints2d_list)
        smpl_data = self.estimate_smpl(
            keypoints3d=keypoints3d, init_smpl_data=init_smpl_data)
        return keypoints2d_list, keypoints3d, smpl_data

    def estimate_keypoints2d(
        self,
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[List[str]]] = None,
    ) -> List[Keypoints]:
        """Estimate keypoints2d in a top-down way.

        Args:
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
            img_paths (Union[None, List[List[str]]], optional):
                A nested list of image paths, in shape
                [n_view, n_frame]. Defaults to None.

        Returns:
            List[Keypoints]:
                A list of keypoints2d instances.
        """
        self.logger.info('Estimating keypoints2d.')
        input_list = [img_arr, img_paths]
        input_count = 0
        for input_instance in input_list:
            if input_instance is not None:
                input_count += 1
        if input_count > 1:
            self.logger.error('Redundant input!\n' +
                              'Please offer only one between' +
                              ' img_arr, img_paths.')
            raise ValueError
        ret_list = []
        for view_index in range(img_arr.shape[0]):
            if img_arr is not None:
                view_img_arr = img_arr[view_index]
                bbox_list = self.bbox_detector.infer_array(
                    image_array=view_img_arr,
                    disable_tqdm=(not self.verbose),
                    multi_person=False)
                kps2d_list, _, _ = self.kps2d_estimator.infer_array(
                    image_array=view_img_arr,
                    bbox_list=bbox_list,
                    disable_tqdm=(not self.verbose),
                )
            else:
                bbox_list = self.bbox_detector.infer_frames(
                    frame_path_list=img_paths,
                    disable_tqdm=(not self.verbose),
                    multi_person=True,
                    load_batch_size=self.load_batch_size)
                kps2d_list, _, _ = self.kps2d_estimator.infer_frames(
                    frame_path_list=img_paths,
                    bbox_list=bbox_list,
                    disable_tqdm=(not self.verbose),
                    load_batch_size=self.load_batch_size)
            if len(kps2d_list) == 1 and \
                    len(kps2d_list[0]) == 1 and \
                    kps2d_list[0][0] is None:
                kps2d_list = [[]]
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            ret_list.append(keypoints2d)
        return ret_list

    def estimate_keypoints3d(self, cam_param: List[FisheyeCameraParameter],
                             keypoints2d_list: List[Keypoints]) -> Keypoints:
        """Estimate keypoints3d by triangulation and optimizers if exists.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            keypoints2d_list (List[Keypoints]):
                A list of Keypoints2d, in same mask and convention,
                and the time axis are aligned.

        Returns:
            Keypoints: A keypoints3d Keypoints instance.
        """
        self.logger.info('Estimating keypoints3d.')
        # prepare input np.ndarray
        kps_arr_list = []
        mask_list = []
        default_keypoints2d = None
        for keypoints2d in keypoints2d_list:
            if keypoints2d is not None:
                default_keypoints2d = keypoints2d.clone()
                default_keypoints2d.set_keypoints(
                    np.zeros_like(default_keypoints2d.get_keypoints()))
                default_keypoints2d.set_mask(
                    np.zeros_like(default_keypoints2d.get_mask()))
                break
        if default_keypoints2d is None:
            self.logger.error('No one has been detected in any view.')
            raise AttributeError
        for keypoints2d in keypoints2d_list:
            if keypoints2d is None:
                keypoints2d = default_keypoints2d
            if keypoints2d.dtype != 'numpy':
                keypoints2d = keypoints2d.to_numpy()
            kps_arr_list.append(keypoints2d.get_keypoints()[:, 0, ...])
            mask_list.append(keypoints2d.get_mask()[:, 0, ...])
        mview_kps2d_arr = np.asarray(kps_arr_list)
        mview_mask = np.asarray(mask_list)
        mview_mask = np.expand_dims(mview_mask, -1)
        # select camera
        cam_indexes = self.select_camera(cam_param, mview_kps2d_arr,
                                         mview_mask)
        self.triangulator.set_cameras(cam_param)
        selected_triangulator = self.triangulator[cam_indexes]
        mview_kps2d_arr = mview_kps2d_arr[np.asarray(cam_indexes), ...]
        triangulate_mask = mview_mask[np.asarray(cam_indexes), ...]
        # cascade point selectors
        self.logger.info('Selecting points.')
        if self.final_selectors is not None:
            for selector in self.final_selectors:
                triangulate_mask = selector.get_selection_mask(
                    points=mview_kps2d_arr, init_points_mask=triangulate_mask)
        kps3d_arr = selected_triangulator.triangulate(
            points=mview_kps2d_arr, points_mask=triangulate_mask)
        kps3d_arr = np.concatenate(
            (kps3d_arr, np.ones_like(kps3d_arr[..., 0:1])), axis=-1)
        kps3d_arr = np.expand_dims(kps3d_arr, axis=1)
        kps3d_mask = np.sum(mview_mask, axis=(0, 1), keepdims=False)
        kps3d_mask = np.sign(np.abs(kps3d_mask))
        if kps3d_mask.shape[-1] == 1:
            kps3d_mask = kps3d_mask[..., 0]
        keypoints3d = Keypoints(
            dtype='numpy',
            kps=kps3d_arr,
            mask=kps3d_mask,
            convention=default_keypoints2d.get_convention())
        optim_kwargs = dict(
            mview_kps2d=np.expand_dims(mview_kps2d_arr, axis=2),
            mview_kps2d_mask=np.expand_dims(triangulate_mask, axis=2))
        if self.kps3d_optimizers is not None:
            for optimizer in self.kps3d_optimizers:
                if hasattr(optimizer, 'triangulator'):
                    optimizer.triangulator = selected_triangulator
                keypoints3d = optimizer.optimize_keypoints3d(
                    keypoints3d, **optim_kwargs)
        return keypoints3d

    def estimate_smpl(self,
                      keypoints3d: Keypoints,
                      init_smpl_data: Union[None, SMPLData] = None,
                      return_joints: bool = False,
                      return_verts: bool = False) -> SMPLData:
        """Estimate smpl parameters according to keypoints3d.

        Args:
            keypoints3d (Keypoints):
                A keypoints3d Keypoints instance, with only one person
                inside. This method will take the person at
                keypoints3d.get_keypoints()[:, 0, ...] to run smplify.
            init_smpl_dict (dict, optional):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
                Defaults to an empty dict.
            return_joints (bool, optional):
                Whether to return joints. Defaults to False.
            return_verts (bool, optional):
                Whether to return vertices. Defaults to False.

        Returns:
            SMPLData:
                Smpl data of the person.
        """
        self.logger.info('Estimating SMPL.')
        working_convention = self.smplify.body_model.keypoint_convention
        keypoints3d = convert_keypoints(
            keypoints=keypoints3d, dst=working_convention)
        keypoints3d = keypoints3d.to_tensor(device=self.smplify.device)
        kps3d_tensor = keypoints3d.get_keypoints()[:, 0, :, :3].float()
        kps3d_conf = keypoints3d.get_mask()[:, 0, ...]

        # load init smpl data
        if init_smpl_data is not None:
            init_smpl_dict = init_smpl_data.to_tensor_dict(
                device=self.smplify.device)
        else:
            init_smpl_dict = {}

        # build and run
        kp3d_mse_input = build_handler(
            dict(
                type='Keypoint3dMSEInput',
                keypoints3d=kps3d_tensor,
                keypoints3d_conf=kps3d_conf,
                keypoints3d_convention=working_convention,
                handler_key='keypoints3d_mse'))
        kp3d_llen_input = build_handler(
            dict(
                type='Keypoint3dLimbLenInput',
                keypoints3d=kps3d_tensor,
                keypoints3d_conf=kps3d_conf,
                keypoints3d_convention=working_convention,
                handler_key='keypoints3d_limb_len'))

        registrant_output = self.smplify(
            input_list=[kp3d_mse_input, kp3d_llen_input],
            init_param_dict=init_smpl_dict,
            return_joints=return_joints,
            return_verts=return_verts)

        if self.smpl_data_type == 'smplx':
            smpl_data = SMPLXData()
        elif self.smpl_data_type == 'smpl':
            smpl_data = SMPLData()

        smpl_data.from_param_dict(registrant_output)

        if return_joints:
            smpl_data['joints'] = registrant_output['joints']
        if return_verts:
            smpl_data['vertices'] = registrant_output['vertices']

        return smpl_data

    def select_camera(self, cam_param: List[FisheyeCameraParameter],
                      points: np.ndarray,
                      points_mask: np.ndarray) -> List[int]:
        """Use cam_pre_selector to filter bad points, use reprojection error of
        the good points to select good cameras.

        Args:
            cam_param (List[FisheyeCameraParameter]):
                A list of FisheyeCameraParameter instances.
            points (np.ndarray):
                Multiview points2d, in shape [n_view, n_frame, n_kps, 3].
                Point scores at the last dim.
            points_mask (np.ndarray):
                Multiview points2d mask,
                in shape [n_view, n_frame, n_kps, 1].

        Returns:
            List[int]: A list of camera indexes.
        """
        if self.cam_selector is not None:
            self.logger.info('Selecting cameras.')
            if self.cam_pre_selector is not None:
                self.logger.info('Using pre-selector for camera selection.')
                pre_mask = self.cam_pre_selector.get_selection_mask(
                    points=points, init_points_mask=points_mask)
            else:
                pre_mask = points_mask.copy()
            self.triangulator.set_cameras(cam_param)
            self.cam_selector.triangulator = self.triangulator
            selected_camera_indexes = self.cam_selector.get_camera_indexes(
                points=points, init_points_mask=pre_mask)
            self.logger.info(f'Selected cameras: {selected_camera_indexes}')
        else:
            self.logger.warning(
                'The estimator api instance has no cam_selector,' +
                ' all the cameras will be returned.')
            selected_camera_indexes = [idx for idx in range(len(cam_param))]
        return selected_camera_indexes
