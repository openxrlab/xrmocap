import logging
import numpy as np
from typing import List, Tuple, Union

from xrmocap.utils.mvpose_utils import get_distance
from .dfs_matching import DFSMatching


class DFSHomoMatching(DFSMatching):

    def __init__(self,
                 total_depth: int,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Depth first search for matching, used for academic dataset.

        Args:
            total_depth (int):
                Total depth. Typically it is number of cameras.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(total_depth=total_depth, logger=logger)

    def recursive_step(self, curr_depth: int, mview_kps2d: np.ndarray) -> None:
        for actor_idx, curr_points in enumerate(self.points[curr_depth]):
            if get_distance(
                    self.homo_anchor,
                    curr_points) > self.homo_thres or self.mark_mat[
                        curr_depth,
                        actor_idx] > 0 or actor_idx > 0 and self.nan_mat[
                            curr_depth, actor_idx - 1] > 0:
                continue
            # bottom not reached, step again
            elif curr_depth < self.total_depth:
                # Generate the single triangulate point set
                self.matched_human_idx.append(actor_idx)
                self.matched_human_kps.append(mview_kps2d[curr_depth,
                                                          actor_idx])
                self.recursive_step(
                    curr_depth=curr_depth + 1, mview_kps2d=mview_kps2d)
                self.matched_human_idx.pop(-1)
                self.matched_human_kps.pop(-1)
            # bottom reached, stop and return
            elif curr_depth == self.total_depth:
                self.matched_human_idx.append(actor_idx)
                self.matched_human_kps.append(mview_kps2d[curr_depth,
                                                          actor_idx])

                matched_human_idx_np = np.array(self.matched_human_idx)
                matched_human_kps2d_np = np.array(self.matched_human_kps)
                n_human_cam = (np.isnan(matched_human_kps2d_np).sum(
                    axis=1)[:, 0] < 1).sum()
                if n_human_cam >= 2:  # minimum two cameras are available
                    self.matching_results[0] += [
                        matched_human_idx_np,
                    ]
                    self.matching_results[1] += [
                        matched_human_kps2d_np,
                    ]
                    self.matching_results[2] += [
                        n_human_cam,
                    ]
                self.matched_human_idx.pop(-1)
                self.matched_human_kps.pop(-1)

    def __call__(self, mview_kps2d: np.ndarray, mark_mat: np.ndarray,
                 nan_mat: np.ndarray, init_matched_human_idx: List[int],
                 init_matched_human_kps: List[np.ndarray],
                 homo_thres: np.float64, homo_anchor: np.ndarray,
                 points: np.ndarray, **kwargs) -> Tuple[list, list, list]:
        """Compute method of Matching instance. Giving multi-view kps2d, it
        will return the match results.

        Args:
            mview_kps2d (np.ndarray): Multi-view keypoints 2d array,
                in shape [n_view, n_person, n_kps2d, 2].
            mark_mat (np.ndarray): Init matrix for observation, 1 indicates
                person matched in sight
                in shape [n_view, n_person]
            nan_mat (np.ndarray): Matrix for nan, in shape
                [n_view, n_person], True indicates observation
                there is np.nan
            init_matched_human_idx (List[int]): Init list of matched
                human index.
            init_matched_human_kps (List[np.ndarray]): Init list of matched
                human kps array.
            homo_thres (np.float64): Threshold of current 3d human.
            homo_anchor (np.ndarray): Anchor of current 3d human.
            points (np.ndarray): H projected homography points (left feet).

        Returns:
            Tuple[list, list, list]:
                matched_kps2d (list): Matched human keypoints.
                matched_human_idx (list): Matched human index.
                matched_observation (list): Matched human observed
                    camera number.
        """
        self.matching_results = [[], [], []]
        self.mark_mat = mark_mat
        self.nan_mat = nan_mat
        self.matched_human_idx = init_matched_human_idx
        self.matched_human_kps = init_matched_human_kps

        self.homo_thres = homo_thres
        self.homo_anchor = homo_anchor
        self.points = points
        curr_depth = len(init_matched_human_idx)
        self.recursive_step(curr_depth=curr_depth, mview_kps2d=mview_kps2d)
        return self.matching_results
