# yapf: disable
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple, Union, overload
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.human_perception.builder import (
    MMdetDetector, MMposeTopDownEstimator, build_detector,
)
from xrmocap.io.image import (
    get_n_frame_from_mview_src, load_clip_from_mview_src,
)
from xrmocap.model.registrant.builder import SMPLify
from xrmocap.ops.top_down_association.builder import (
    MvposeAssociator, build_top_down_associator,
)
from xrmocap.ops.triangulation.builder import (
    BaseTriangulator, build_triangulator,
)
from xrmocap.ops.triangulation.point_selection.builder import (
    BaseSelector, build_point_selector,
)
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_num,
)
from xrmocap.transform.keypoints3d.optim.builder import BaseOptimizer
from .mperson_smpl_estimator import MultiPersonSMPLEstimator

# yapf: enable


class MultiViewMultiPersonTopDownEstimator(MultiPersonSMPLEstimator):
    """Api for estimating keypoints3d and smpl in a multi-view multi-person
    scene, using optimization-based top-down method."""

    def __init__(self,
                 bbox_thr: float,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 associator: Union[dict, MvposeAssociator],
                 smplify: Union[dict, SMPLify],
                 triangulator: Union[dict, BaseTriangulator],
                 point_selectors: List[Union[dict, BaseSelector, None]] = None,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 pred_kps3d_convention: str = 'coco',
                 load_batch_size: int = 10,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialization of the class.

        Args:
            bbox_thr (float):
                The threshold of the bbox2d.
            work_dir (str):
                Path to the folder for running the api.
                No file in work_dir will be modified or added by
                MultiViewMultiPersonTopDownEstimator.
            bbox_detector (Union[dict, MMdetDetector]):
                A human bbox_detector or its config.
            kps2d_estimator (Union[dict, MMposeTopDownEstimator]):
                A top-down kps2d estimator or its config.
            associator (Union[dict, MvposeAssociator]):
                A MvposeAssociator instance or its config.
            smplify (Union[dict, SMPLify]):
                A SMPLify instance or its config.
            triangulator (Union[dict, BaseTriangulator]):
                A triangulator or its config.
            point_selectors (List[Union[dict, BaseSelector, None]], optional):
                A point selector or its config. If it's given, points
                will be selected before triangulation.
                Defaults to None.
            kps3d_optimizers (Union[
                    List[Union[BaseOptimizer, dict]], None], optional):
                A list of keypoints3d optimizers or their configs. If given,
                keypoints3d will be optimized by the cascaded final optimizers.
                Defaults to None.
            pred_kps3d_convention (str, optional): Defaults to 'coco'.
            load_batch_size (int, optional):
                How many frames are loaded at the same time. Defaults to 10.
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

        if isinstance(triangulator, dict):
            triangulator['logger'] = logger
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator

        if point_selectors is None:
            self.point_selectors = None
        else:
            self.point_selectors = []
            for selector in point_selectors:
                if isinstance(selector, dict):
                    selector['logger'] = logger
                    selector = build_point_selector(selector)
                self.point_selectors.append(selector)

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
    ) -> Tuple[Keypoints, List[SMPLData]]:
        """Run mutli-view multi-person topdown estimator once. run() needs one
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
            Tuple[Keypoints, List[SMPLData]]:
                A keypoints3d, a list of SMPLData.
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
        self.associator.set_cameras(cam_param)
        self.triangulator.set_cameras(cam_param)
        if self.point_selectors is not None:
            for selector in self.point_selectors:
                if hasattr(selector, 'triangulator'):
                    selector.triangulator.set_cameras(cam_param)
        n_view = len(cam_param)
        n_kps = get_keypoint_num(convention=self.pred_kps3d_convention)
        pred_kps3d = np.zeros((n_frame, 1, n_kps, 4))
        association_results = [[] for _ in range(n_frame)]
        selected_keypoints2d = []
        max_identity = 0
        for start_idx in tqdm(
                range(0, n_frame, self.load_batch_size),
                disable=not self.verbose):
            end_idx = min(n_frame, start_idx + self.load_batch_size)
            mview_batch_arr = load_clip_from_mview_src(
                start_idx=start_idx,
                end_idx=end_idx,
                img_arr=img_arr,
                img_paths=img_paths,
                video_paths=video_paths,
                logger=self.logger)
            # Estimate bbox2d and keypoints2d
            bbox2d_list, keypoints2d_list = self.estimate_perception2d(
                mview_batch_arr)
            for sub_fidx in range(end_idx - start_idx):
                sframe_bbox2d_list = []
                sframe_keypoints2d_list = []
                for view_idx in range(n_view):
                    sview_kps2d_idx = []
                    for idx, bbox2d in enumerate(
                            bbox2d_list[view_idx][sub_fidx]):
                        if bbox2d[-1] > self.bbox_thr:
                            sview_kps2d_idx.append(idx)
                    sview_kps2d_idx = np.array(sview_kps2d_idx)
                    if len(sview_kps2d_idx) > 0:
                        sframe_bbox2d_list.append(
                            torch.tensor(bbox2d_list[view_idx][sub_fidx])
                            [sview_kps2d_idx])
                        mframe_keypoints2d = keypoints2d_list[view_idx]
                        keypoints2d = Keypoints(
                            kps=mframe_keypoints2d.get_keypoints()[
                                sub_fidx:sub_fidx + 1, sview_kps2d_idx],
                            mask=mframe_keypoints2d.get_mask()[
                                sub_fidx:sub_fidx + 1, sview_kps2d_idx],
                            convention=mframe_keypoints2d.get_convention(),
                            logger=self.logger)
                        if keypoints2d.get_convention(
                        ) != self.pred_kps3d_convention:
                            keypoints2d = convert_keypoints(
                                keypoints=keypoints2d,
                                dst=self.pred_kps3d_convention,
                                approximate=True)
                        sframe_keypoints2d_list.append(keypoints2d)
                    else:
                        sframe_bbox2d_list.append(torch.tensor([]))
                        sframe_keypoints2d_list.append(
                            Keypoints(
                                kps=np.zeros((1, 1, n_kps, 3)),
                                mask=np.zeros((1, 1, n_kps)),
                                convention=self.pred_kps3d_convention))
                # Establish cross-frame and cross-person associations
                sframe_association_results, predict_keypoints3d, identities = \
                    self.associator.associate_frame(
                        # Dimension definition varies between
                        # cv2 images and tensor images.
                        mview_img_arr=mview_batch_arr[:, sub_fidx].transpose(
                            0, 3, 1, 2),
                        mview_bbox2d=sframe_bbox2d_list,
                        mview_keypoints2d=sframe_keypoints2d_list,
                        affinity_type='geometry_mean'
                    )
                for p_idx in range(len(sframe_association_results)):
                    # Triangulation, one associated person per time
                    identity = identities[p_idx]
                    associate_idxs = sframe_association_results[p_idx]
                    tri_kps2d = np.zeros((n_view, n_kps, 3))
                    tri_mask = np.zeros((n_view, n_kps, 1))
                    for view_idx in range(n_view):
                        kps2d_idx = associate_idxs[view_idx]
                        if not np.isnan(kps2d_idx):
                            tri_kps2d[view_idx] = sframe_keypoints2d_list[
                                view_idx].get_keypoints()[0, int(kps2d_idx)]
                            tri_mask[view_idx, :, 0] = sframe_keypoints2d_list[
                                view_idx].get_mask()[0, int(kps2d_idx)]
                    if self.point_selectors is not None:
                        for selector in self.point_selectors:
                            tri_mask = selector.get_selection_mask(
                                points=tri_kps2d, init_points_mask=tri_mask)
                    kps3d = self.triangulator.triangulate(tri_kps2d, tri_mask)
                    if identity > max_identity:
                        n_identity = identity - max_identity
                        pred_kps3d = np.concatenate(
                            (pred_kps3d,
                             np.zeros((n_frame, n_identity, n_kps, 4))),
                            axis=1)
                        max_identity = identity
                    pred_kps3d[start_idx + sub_fidx,
                               identity] = np.concatenate(
                                   (kps3d, np.ones_like(kps3d[:, 0:1])),
                                   axis=-1)
                for identity in sorted(identities):
                    index = identities.index(identity)
                    association_results[start_idx + sub_fidx].append(
                        sframe_association_results[index])
                selected_keypoints2d.append(sframe_keypoints2d_list)
        # Convert array to keypoints instance
        pred_keypoints3d = Keypoints(
            dtype='numpy',
            kps=pred_kps3d,
            mask=pred_kps3d[..., -1] > 0,
            convention=self.pred_kps3d_convention,
            logger=self.logger)
        # Save keypoints2d
        selected_keypoints2d_list = []
        mview_person_id = [[] for _ in range(n_view)]
        for view_idx in range(n_view):
            pred_kps2d = np.zeros((n_frame, 1, n_kps, 3))
            max_n_kps2d = 1
            for frame_idx, sframe_keypoints2d in enumerate(
                    selected_keypoints2d):
                kps2d = sframe_keypoints2d[view_idx].get_keypoints()
                n_kps2d = sframe_keypoints2d[view_idx].get_person_number()
                if n_kps2d > max_n_kps2d:
                    pred_kps2d = np.concatenate(
                        (pred_kps2d,
                         np.zeros(
                             (n_frame, (n_kps2d - max_n_kps2d), n_kps, 3))),
                        axis=1)
                    max_n_kps2d = n_kps2d
                pred_kps2d[frame_idx, :n_kps2d] = kps2d[0]
                mview_person_id[view_idx].append(np.array(range(n_kps2d)))
            selected_keypoints2d_list.append(
                Keypoints(
                    kps=pred_kps2d,
                    mask=pred_kps2d[..., -1] > 0,
                    convention=self.pred_kps3d_convention))

        optim_kwargs = dict(
            keypoints2d=selected_keypoints2d_list,
            mview_person_id=mview_person_id,
            matched_list=association_results,
            cam_params=cam_param)
        # Optimizing keypoints3d
        pred_keypoints3d = self.optimize_keypoints3d(pred_keypoints3d,
                                                     **optim_kwargs)
        # Fitting SMPL model
        smpl_data_list = self.estimate_smpl(keypoints3d=pred_keypoints3d)

        return pred_keypoints3d, smpl_data_list

    def estimate_perception2d(
        self, img_arr: Union[None, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Keypoints]]:
        """Estimate bbox2d and keypoints2d.

        Args:
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
        Returns:
            Tuple[List[np.ndarray], List[Keypoints]]:
                A list of bbox2d, and a list of keypoints2d instances.
        """
        mview_bbox2d_list = []
        mview_keypoints2d_list = []
        for view_index in range(img_arr.shape[0]):
            bbox2d_list = self.bbox_detector.infer_array(
                image_array=img_arr[view_index],
                disable_tqdm=(not self.verbose),
                multi_person=True)
            kps2d_list, _, bbox2d_list = self.kps2d_estimator.infer_array(
                image_array=img_arr[view_index],
                bbox_list=bbox2d_list,
                disable_tqdm=(not self.verbose))
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            mview_bbox2d_list.append(bbox2d_list)
            mview_keypoints2d_list.append(keypoints2d)
        return mview_bbox2d_list, mview_keypoints2d_list
