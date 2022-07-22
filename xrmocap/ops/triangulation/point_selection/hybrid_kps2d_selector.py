# yapf: disable
import logging
import numpy as np
from typing import List, Union
from xrprimer.ops.triangulation.base_triangulator import BaseTriangulator

from xrmocap.matching.pictorial import (
    get_conns, get_struct, infer_kps3d_max_product,
)
from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.transform.convention.keypoints_convention import (
    get_keypoint_idx, get_keypoint_num,
)
from xrmocap.utils.triangulation_utils import prepare_triangulate_input
from .base_selector import BaseSelector

# yapf: enable


class HybridKps2dSelector(BaseSelector):

    def __init__(self,
                 triangulator: Union[BaseTriangulator, dict],
                 distribution: dict,
                 verbose: bool = True,
                 ignore_kps_name: List[str] = None,
                 convention: str = 'coco',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Select points according to camera reprojection error. This selector
        will disable the worst cameras according to one reprojection result.

        Args:
            triangulator (Union[BaseSelector, dict]):
                Triangulator for reprojection error calculation.
                An instance or config dict.
            distribution (dict): Bone constraints. Defaults to None.
            verbose (bool, optional):
                Whether to log info like valid views stats.
                Defaults to True.
            ignore_kps_name (List[str], optional): The name of ignore kps.
                Defaults to None.
            convention (str, optional): Defaults to 'coco'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(verbose=verbose, logger=logger)
        if isinstance(triangulator, dict):
            self.triangulator = build_triangulator(triangulator)
        else:
            self.triangulator = triangulator
        self.distribution = distribution
        self.ignore_kps_name = ignore_kps_name
        self.convention = convention
        self.ignore_indexes = []
        for name in self.ignore_kps_name:
            index = get_keypoint_idx(name=name, convention=self.convention)
            self.ignore_indexes.append(index)

    def get_selection_mask(
        self,
        points: Union[np.ndarray, list, tuple],
        init_points_mask: Union[np.ndarray, list, tuple],
    ) -> np.ndarray:
        """Get a new selection mask from points and init_points_mask. This
        selector will loop triangulate points, disable the one camera with
        largest reprojection error, and loop again until there are
        self.target_camera_number left.

        Args:
            points (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of points2d, in shape
                [n_view, n_points, 2+1].
            init_points_mask (Union[np.ndarray, list, tuple]):
                An ndarray or a nested list of mask, in shape
                [n_view, n_points, 1].
                If points_mask[index] == 1, points[index] is valid
                for triangulation, else it is ignored.
                If points_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
                Defaults to None.

        Returns:
            np.ndarray:
                An ndarray or a nested list of mask, in shape
                [n_view, n_points, 1].
        """
        points, init_points_mask = prepare_triangulate_input(
            camera_number=len(points),
            points=points,
            points_mask=init_points_mask,
            logger=self.logger)
        selected_candidates, candidates2camid = self.get_candidates_indexes(
            kps2d=points[..., :2],
            init_kps2d_mask=init_points_mask,
            kps2d_conf=points[..., 2:3])
        points2d_mask = np.zeros_like(init_points_mask)
        for kps_idx, candidates_idx in zip(
                range(selected_candidates.shape[0]), selected_candidates):
            selected_cam_0, selected_cam_1 = candidates2camid[candidates_idx]
            points2d_mask[[selected_cam_0, selected_cam_1], kps_idx] = 1

        return points2d_mask

    def get_candidates_indexes(
            self, kps2d: np.ndarray, init_kps2d_mask: np.ndarray,
            kps2d_conf: np.ndarray) -> Union[np.ndarray, dict]:
        """Get an ndarray of kps2d indexes. This selector will loop triangulate
        kps2d, enable the kps2d index.

        Args:
            kps2d (np.ndarray):
                An ndarray of points2d, in shape [n_view, n_kps2d, 2].
            init_kps2d_mask (np.ndarray):
                An ndarray of mask, in shape [n_view, ..., 1].
                If kps2d_mask[index] == 1, kps2d[index] is valid
                for triangulation, else it is ignored.
                If kps2d_mask[index] == np.nan, the whole pair will
                be ignored and not counted by any method.
            kps2d_conf (np.ndarray): An ndarray of points2d confidence,
                in shape [n_view, n_kps2d, 1].

        Returns:
            selected_cand_idx (np.ndarray):
                An ndarray of sorted candidates indexes.
            candidates2camid (dict):
                The camera id selected for each triangulation process.
        """
        n_mview_person = 0
        n_kps2d = kps2d.shape[-2]
        sub_imgid2cam = []
        for i, mask in enumerate(init_kps2d_mask):
            if mask.all():
                n_mview_person += 1
                sub_imgid2cam.append(i)
        # step1: use 2D joint of person to triangulate the 3D
        # person's 17 3D joints candidates in shape(n_kps, C^2_n, 3)
        candidates = np.zeros(
            (n_kps2d, n_mview_person * (n_mview_person - 1) // 2, 3))
        cnt = 0
        candidates2camid = {}
        for i in range(n_mview_person):
            for j in range(i + 1, n_mview_person):
                candidates2camid[cnt] = [sub_imgid2cam[i], sub_imgid2cam[j]]
                selected_kps2d = np.stack(
                    (kps2d[sub_imgid2cam[i]], kps2d[sub_imgid2cam[j]]), axis=0)
                kps3d_ij = self.triangulator[(sub_imgid2cam[i],
                                              sub_imgid2cam[j])].triangulate(
                                                  points=selected_kps2d)
                candidates[:, cnt] += kps3d_ij
                cnt += 1
        unary = self.get_unary(kps2d, kps2d_conf, sub_imgid2cam, candidates)

        # step2: use the max-product algorithm to inference the 3d joint
        # change the coco order
        selected_kps = []
        n_kps = get_keypoint_num(self.convention)
        for i in range(n_kps):
            if i not in self.ignore_indexes:
                selected_kps.append(i)
        n_kps = len(selected_kps)
        candidates = np.array(candidates)[selected_kps]
        unary = unary[selected_kps]
        conns = get_conns(n_kps)
        # construct pictorial model
        limb = get_struct(conns, self.distribution)
        selected_cand_idx = infer_kps3d_max_product(unary, limb, candidates)

        ret_selected_cand_idx = np.zeros(
            n_kps2d, dtype=selected_cand_idx.dtype)
        ret_selected_cand_idx[selected_kps] = selected_cand_idx
        # Just make visualize beauty not real ear and eye
        ret_selected_cand_idx[self.ignore_indexes] = ret_selected_cand_idx[0]
        return ret_selected_cand_idx, candidates2camid

    def get_unary(self,
                  mview_kps2d: np.ndarray,
                  mview_kps2d_conf: np.ndarray,
                  sub_imgid2cam: np.ndarray,
                  candidates: np.ndarray,
                  use_heatmap=False,
                  heatmap: np.ndarray = None) -> np.ndarray:
        """Get the probability of candidate for each kps.

        Args:
            mview_kps2d (np.ndarray): An ndarray of points2d,
                in shape [n_view, n_kps2d, 2].
            mview_kps2d_conf (np.ndarray): An ndarray of points2d confidence,
                in shape [n_view, n_kps2d, 1].
            sub_imgid2cam (np.ndarray): Person id to camera id.
            candidates (np.ndarray): The candidates of kps3d, in shape
                [n_kps3d, n_candidates, 3].
            use_heatmap (bool, optional): Defaults to False.
            heatmap (np.ndarray, optional): The heatmap data. Defaults to None.

        Raises:
            NotImplementedError: Other n_kps3d have not been supported yet.
            NotImplementedError: Other n_kps3d have not been supported yet.

        Returns:
            np.ndarray: the probability of candidate for each kps, in shape
                [n_kps3d, n_candidates].
        """
        # get the unary of 3D candidates
        n_kps2d = len(candidates)
        n_point = len(candidates[0])
        n_cameras = mview_kps2d.shape[0]
        unary = np.ones((n_kps2d, n_point))
        camera_parameter = {
            'P': np.zeros((n_cameras, 3, 4)),
            'K': np.zeros((n_cameras, 3, 3)),
            'RT': np.zeros((n_cameras, 3, 4)),
        }
        for i, cam_param in enumerate(self.triangulator.camera_parameters):
            camera_parameter['K'][i] = cam_param.get_intrinsic(k_dim=3)
            camera_parameter['RT'][i] = np.concatenate(
                (np.asarray(cam_param.extrinsic_r),
                 np.asarray(cam_param.extrinsic_t)[:, np.newaxis]),
                axis=1)
            # compute projection matrix
            proj_mat = camera_parameter['K'][i] @ camera_parameter['RT'][i]
            camera_parameter['P'][i] = proj_mat
        # project the 3d point to each view to get the 2d points
        for pid in sub_imgid2cam:
            Pi = camera_parameter['P'][pid]
            if n_kps2d != 17:
                self.logger.error('Other n_kps3d have not been supported yet.')
            if not use_heatmap:
                kps2d = mview_kps2d[pid]
                kps2d_conf = mview_kps2d_conf[pid]
            else:
                heatmap = heatmap['heatmap_data']
                crop = np.array(heatmap['heatmap_bbox'])
            kps3d = candidates.reshape(-1, 3).T
            kps3d_homo = np.vstack(
                [kps3d, np.ones(kps3d.shape[-1]).reshape(1, -1)])
            kps2d_homo = (Pi @ kps3d_homo).T.reshape(n_kps2d, -1, 3)
            proj_kps2d = kps2d_homo[..., :2] / (
                kps2d_homo[..., 2].reshape(n_kps2d, -1, 1) + 10e-6)

            if not use_heatmap:
                for kps_idx, kp2d in enumerate(kps2d):
                    for j, _ in enumerate(candidates[kps_idx]):
                        proj_kp2d = proj_kps2d[kps_idx, j]
                        # we use gaussian to approx the heatmap
                        if np.isnan(kp2d).any():
                            unary_i = 10e-6
                        else:
                            pixel_distance = ((kp2d - proj_kp2d)**2).sum()
                            unary_i = np.exp(
                                -pixel_distance / 625) * kps2d_conf[kps_idx]
                            unary_i = np.clip(unary_i, 10e-6, 1)

                        unary[kps_idx, j] = unary[kps_idx, j] * unary_i
            else:
                for kps_idx, heatmap_ in enumerate(heatmap):
                    for j, _ in enumerate(candidates[kps_idx]):
                        proj_kp2d = proj_kps2d[kps_idx, j]
                        kp2d_in_heatmap = proj_kp2d - np.array(
                            [crop[0], crop[1]])

                        if kp2d_in_heatmap[0] > heatmap_.shape[1] or \
                            kp2d_in_heatmap[0] < 0 or kp2d_in_heatmap[
                                1] > heatmap_.shape[0] or kp2d_in_heatmap[
                                    1] < 0:
                            unary_i = 10e-6
                        else:
                            unary_i = heatmap_[int(kp2d_in_heatmap[1]),
                                               int(kp2d_in_heatmap[0])]
                        unary[kps_idx, j] = unary[kps_idx, j] * unary_i

        unary = np.log10(unary)
        return unary
