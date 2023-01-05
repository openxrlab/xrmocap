# yapf: disable
import itertools
import logging
import numpy as np
from typing import Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    get_keypoint_idxs_by_part,
)
from xrmocap.transform.limbs import search_limbs
from .base_optimizer import BaseOptimizer

# yapf: enable


class PriorConstraint(BaseOptimizer):
    """This optimization tool sets a bbox and remove 3D keypoints with large
    movement between frames for move outside the bbox.

    The bbox orientation is ignored.
    """

    def __init__(
        self,
        verbose: bool = True,
        bbox_size: list = [2.0, 2.0, 2.0],
        stand_only: bool = False,
        ground_norm: list = [0.0, 1.0, 0.0],
        standing_thr: float = 0.0,
        use_limb_length: bool = False,
        n_max_frame: int = 5,
        diff_thr: float = 0.5,
        update_mask: bool = False,
        logger: Union[None, str, logging.Logger] = None,
    ):
        """Initialization for MovementConstrain optimizor.

        Args:
            verbose (bool, optional):
                Whether to log info. Defaults to True.
            bbox_size (list, optional):
                Edge length of the bounding box.
                Unit should align with input data.
                Defaults to [2.0, 2.0, 2.0].
            stand_only (bool, optional):
                Only apply bbox constraint to standing position.
                Defaults to False.
            ground_norm (list, optional):
                Norm of the ground in the world coordinate.
                This is used to approximate whether the person
                is in the standing position.
                Defaults to [0.0, 1.0, 0.0].
            standing_thr (float, optional):
                Body center height threshold for standing
                position in the world coordinate.
                Unit should align with input data.
                Defaults to 0.0.
            use_limb_length (bool, optional):
                Whether to use bone length prior constraint
                between frames.
                Defaults to False.
            n_max_frame (int, optional):
                Max number of frames used to constrain
                the bone length. Defaults to 5.
            diff_thr (float, optional):
                Threshold for the changes of bone length
                between frames in ratio. Defaults to 0.5.
            update_mask (bool, optional):
                Update keypoints mask for those keypoints
                set to nan in this optimizer. Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Returns:
            Keypoints: Optimized keypoints3d
        """
        super().__init__(verbose=verbose, logger=logger)

        self.stand_only = stand_only
        self.ground_norm = np.array(ground_norm)
        self.standing_thr = standing_thr
        self.use_limb_length = use_limb_length
        self.n_max_frame = n_max_frame
        self.diff_thr = diff_thr
        self.offset = np.array(bbox_size) / 2.0
        self.update_mask = update_mask

    def optimize_keypoints3d(self, keypoints3d: Keypoints,
                             **kwargs: dict) -> Keypoints:

        # get kps3d array
        keypoints3d_optim = keypoints3d.clone()
        kps3d = keypoints3d_optim.get_keypoints()[..., :3]
        kps3d_mask = keypoints3d_optim.get_mask()
        n_frame, n_max_person, n_kps, _ = kps3d.shape

        kps3d_optim = np.full((n_frame, n_max_person, n_kps, 4), np.nan)

        # get body center from shoulder and hip keypoints
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=keypoints3d_optim.get_convention())
        hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=keypoints3d_optim.get_convention())
        hip_shoulder_idxs = shoulder_keypoint_idxs + hip_keypoint_idxs

        if self.use_limb_length:
            limb_idxs, _ = search_limbs(
                data_source=keypoints3d.get_convention())
            limb_idxs = sorted(limb_idxs['body'])
            self.limb_idxs = np.array(
                list(x for x, _ in itertools.groupby(limb_idxs)))

        for frame_idx in range(n_frame):
            for person_idx in range(n_max_person):
                kps3d_person = kps3d[frame_idx, person_idx, ...]
                kps3d_person_mask = kps3d_mask[frame_idx, person_idx, ...]

                # skip if the person does not exist in this frame
                if np.isnan(kps3d_person).all():
                    continue
                hip_shoulder_mask = kps3d_person_mask[hip_shoulder_idxs]
                kps3d_hip_shoulder = kps3d_person[hip_shoulder_idxs]
                body_center = kps3d_hip_shoulder[hip_shoulder_mask == 1].mean(
                    axis=0)

                if self.use_limb_length:
                    check_previous_frame = True
                    for i in range(1, self.n_max_frame):
                        if frame_idx - i < 1:
                            break
                        kps3d_person_previous = kps3d[frame_idx - i,
                                                      person_idx, ...]
                        if not np.isnan(kps3d_person_previous).all():
                            limb_len_previous = self._compute_limb_length(
                                kps3d_person_previous)
                            limb_len_curr = self._compute_limb_length(
                                kps3d_person)

                            curr_bone_length_diff = np.abs(limb_len_curr -
                                                           limb_len_previous)
                            per_bone_length_thr = \
                                self.diff_thr * limb_len_previous

                            limb_reset = self.limb_idxs[
                                curr_bone_length_diff > per_bone_length_thr]
                            kps3d_person[limb_reset[:, 0]] = np.nan
                            kps3d_person[limb_reset[:, 1]] = np.nan

                            # check previous frame until all limbs compared
                            # with valid length once or reach the max number
                            # of checking frame
                            check_previous_frame = np.isnan(
                                limb_len_previous).any()
                            if not check_previous_frame:
                                break

                if self.stand_only:
                    height_from_ground = np.dot(
                        np.array(body_center), self.ground_norm)
                    if height_from_ground < self.standing_thr:
                        self.logger.info(
                            f'Person {person_idx} in frmae {frame_idx} '
                            f'is not in a standing pose: '
                            f'{height_from_ground} with '
                            f'threshold {self.standing_thr}.')
                        kps3d_optim[frame_idx,
                                    person_idx, :, :3] = kps3d_person
                        break

                lower_limit = body_center - self.offset
                upper_limit = body_center + self.offset
                kps3d_person[(kps3d_person > upper_limit).any(axis=1)] = np.nan
                kps3d_person[(kps3d_person < lower_limit).any(axis=1)] = np.nan

                kps3d_optim[frame_idx, person_idx, :, :3] = kps3d_person

        # update keypoints3d
        keypoints3d_optim.set_keypoints(kps3d_optim)
        if self.update_mask:
            keypoints3d_optim.set_mask(kps3d_optim[..., -1] > 0)

        return keypoints3d_optim

    def _compute_limb_length(self, keypoints3d):
        kp_src = keypoints3d[self.limb_idxs[:, 0], :3]
        kp_dst = keypoints3d[self.limb_idxs[:, 1], :3]
        limb_vec = kp_dst - kp_src
        limb_length = np.linalg.norm(limb_vec, axis=1)
        return limb_length
