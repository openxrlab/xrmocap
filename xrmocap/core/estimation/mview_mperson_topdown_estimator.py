# yapf: disable
import logging
import numpy as np
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
from xrmocap.transform.keypoints3d.optim.builder import BaseOptimizer
from .mview_mperson_smpl_estimator import MultiViewMultiPersonSMPLEstimator

# yapf: enable


class MultiViewMultiPersonTopDownEstimator(MultiViewMultiPersonSMPLEstimator):

    def __init__(self,
                 work_dir: str,
                 bbox_detector: Union[dict, MMdetDetector],
                 kps2d_estimator: Union[dict, MMposeTopDownEstimator],
                 associator: Union[dict, MvposeAssociator],
                 triangulator: Union[dict, BaseTriangulator],
                 smplify: Union[dict, SMPLify],
                 point_selectors: List[Union[dict, BaseSelector, None]] = None,
                 kps3d_optimizers: Union[List[Union[BaseOptimizer, dict]],
                                         None] = None,
                 load_batch_size: int = 500,
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Api for estimating smpl parameters in a multi-view, single-person
        scene.

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
            point_selectors (List[Union[dict, BaseSelector, None]], optional):
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
        super().__init__(
            work_dir=work_dir,
            smplify=smplify,
            kps3d_optimizers=kps3d_optimizers,
            verbose=verbose,
            logger=logger)
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
    ) -> Tuple[List[Keypoints], SMPLData]:
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
        self.associator.set_cameras(cam_param)
        self.triangulator.set_cameras(cam_param)
        n_view = len(cam_param)
        max_identity = 0
        for start_idx in range(0, n_frame, self.load_batch_size):
            end_idx = min(n_frame, start_idx + self.load_batch_size)
            mview_batch_arr = load_clip_from_mview_src(
                start_idx=start_idx,
                end_idx=end_idx,
                img_arr=img_arr,
                img_paths=img_paths,
                video_paths=video_paths,
                logger=self.logger)
            bbox_list, keypoints2d_list = self.estimate_perception2d(
                mview_batch_arr)
            for sub_fidx in range(end_idx - start_idx):
                sframe_bbox_list = []
                sframe_keypoints2d_list = []
                for view_idx in range(n_view):
                    sframe_bbox_list.append(bbox_list[view_idx][sub_fidx])
                    mframe_keypoints2d = keypoints2d_list[view_idx]
                    keypoints2d = Keypoints(
                        kps=mframe_keypoints2d.get_keypoints()[
                            sub_fidx:sub_fidx + 1, ...],
                        mask=mframe_keypoints2d.get_mask()[sub_fidx:sub_fidx +
                                                           1, ...],
                        convention=mframe_keypoints2d.get_convention(),
                        logger=self.logger)
                    n_kps = keypoints2d.get_keypoints_number()
                    sframe_bbox_list.append(bbox_list[view_idx][sub_fidx])
                    sframe_keypoints2d_list.append(keypoints2d)
                association_results, predict_keypoints3d, identities = \
                    self.associator.associate_frame(
                        mview_img_arr=mview_batch_arr[:, sub_fidx, ...],
                        mview_bbox2d=sframe_bbox_list,
                        mview_keypoints2d=sframe_keypoints2d_list,
                        affinity_type='geometry_mean'
                    )
                for p_idx in range(len(association_results)):
                    identity = identities[p_idx]
                    associate_idxs = association_results[p_idx]
                    tri_kps2d = np.zeros(shape=(n_view, n_kps, 3))
                    tri_mask = np.zeros(shape=(n_view, n_kps))
                    for view_idx in range(n_view):
                        kps2d_idx = associate_idxs[view_idx]
                        if kps2d_idx is not None:
                            tri_kps2d[view_idx] = \
                                sframe_keypoints2d_list[
                                    view_idx].get_keypoints()[
                                        0, kps2d_idx, ...]
                            tri_mask[view_idx] = \
                                sframe_keypoints2d_list[
                                    view_idx].get_mask()[
                                        0, kps2d_idx, ...]
                    if self.point_selectors is not None:
                        for selector in self.point_selectors:
                            tri_mask = selector.get_selection_mask(
                                points=tri_kps2d, init_points_mask=tri_mask)
                    kps3d = self.triangulator.triangulate(tri_kps2d, tri_mask)
                    max_identity = max(max_identity, identity)
                    # TODO: set kps3d to proper location by identity
        keypoints3d_list = [
            Keypoints(kps=kps3d, mask=np.ones_like(kps3d[..., 0]))
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

    def estimate_perception2d(
        self, img_arr: Union[None, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Keypoints]]:
        """Estimate bbox2d and keypoints2d.

        Args:
            img_arr (Union[None, np.ndarray], optional):
                A multi-view image array, in shape
                [n_view, n_frame, h, w, c]. Defaults to None.
            img_paths (Union[None, List[List[str]]], optional):
                A nested list of image paths, in shape
                [n_view, n_frame]. Defaults to None.

        Returns:
            Tuple[List[np.ndarray], List[Keypoints]]:
                A list of bbox2d, and a list of keypoints2d instances.
        """
        self.logger.info('Estimating perception2d.')
        mview_bbox_list = []
        mview_keypoints2d_list = []
        for view_index in range(img_arr.shape[0]):
            bbox_list = self.bbox_detector.infer_array(
                image_array=img_arr[view_index],
                disable_tqdm=(not self.verbose),
                multi_person=True)
            kps2d_list, _, bbox_list = self.kps2d_estimator.infer_array(
                image_array=img_arr[view_index],
                bbox_list=bbox_list,
                disable_tqdm=(not self.verbose),
            )
            keypoints2d = self.kps2d_estimator.get_keypoints_from_result(
                kps2d_list)
            mview_bbox_list.append(bbox_list)
            mview_keypoints2d_list.append(keypoints2d)
        return mview_bbox_list, mview_keypoints2d_list
