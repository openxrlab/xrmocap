# yapf: disable
import torch
from typing import Union

from xrmocap.transform.convention.keypoints_convention import (
    get_keypoint_idxs_by_part,
)
from .smplify import SMPLify

# yapf: enable


class SMPLifyX(SMPLify):
    """Re-implementation of SMPLify-X with extended features."""
    OPTIM_PARAM = SMPLify.OPTIM_PARAM + [
        'left_hand_pose', 'right_hand_pose', 'expression', 'jaw_pose',
        'leye_pose', 'reye_pose'
    ]

    def __set_keypoint_indexes__(self) -> None:
        """Set keypoint indexes to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.

        Returns:
            None
        """
        SMPLify.__set_keypoint_indexes__(self)
        convention = self.body_model.keypoint_convention
        # head keypoints include all facial landmarks
        self.face_keypoint_idxs = get_keypoint_idxs_by_part(
            'head', convention=convention)

        left_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'left_hand', convention=convention)
        right_hand_keypoint_idxs = get_keypoint_idxs_by_part(
            'right_hand', convention=convention)
        self.hand_keypoint_idxs = [
            *left_hand_keypoint_idxs, *right_hand_keypoint_idxs
        ]

        self.body_keypoint_idxs = get_keypoint_idxs_by_part(
            'body', convention=convention)

        self.shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)

        self.hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)

        self.foot_keypoint_idxs = get_keypoint_idxs_by_part(
            'foot', convention=convention)

    def get_keypoint_weight(self,
                            use_shoulder_hip_only: bool = False,
                            body_weight: float = 1.0,
                            hand_weight: float = 1.0,
                            face_weight: float = 1.0,
                            shoulder_weight: Union[float, None] = None,
                            hip_weight: Union[float, None] = None,
                            foot_weight: Union[float, None] = None,
                            **kwargs) -> torch.Tensor:
        """Get per keypoint weight.

        Args:
            use_shoulder_hip_only (bool, optional):
                Whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
                Defaults to False.
            body_weight (float, optional):
                Weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            hand_weight (float, optional):
                Weight of hands keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            face_weight (float, optional):
                Weight of face keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to 1.0.
            shoulder_weight (float, optional):
                Weight of shoulder keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.
            hip_weight (float, optional):
                Weight of hip keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.
            foot_weight (float, optional):
                Weight of feet keypoints. Body part segmentation
                definition is included in the HumanData convention.
                Defaults to None.

        Returns:
            torch.Tensor: Per keypoint weight tensor of shape (K).
        """
        n_keypoints = self.body_model.get_joint_number()

        # 3rd priority: set body parts weight manually
        # when both body weight and body parts weight set,
        # body parts weight override the body weight

        weight = torch.ones([n_keypoints]).to(self.device)

        # "body": includes "shoulder", "hip" and "foot"
        weight[self.body_keypoint_idxs] = \
            weight[self.body_keypoint_idxs] * body_weight

        if shoulder_weight is not None:
            weight[self.shoulder_keypoint_idxs] = 1.0
            weight[self.shoulder_keypoint_idxs] = \
                weight[self.shoulder_keypoint_idxs] * shoulder_weight

        if hip_weight is not None:
            weight[self.hip_keypoint_idxs] = 1.0
            weight[self.hip_keypoint_idxs] = \
                weight[self.hip_keypoint_idxs] * hip_weight

        if foot_weight is not None:
            weight[self.foot_keypoint_idxs] = 1.0
            weight[self.foot_keypoint_idxs] = \
                weight[self.foot_keypoint_idxs] * foot_weight

        weight[self.hand_keypoint_idxs] = \
            weight[self.hand_keypoint_idxs] * hand_weight
        weight[self.face_keypoint_idxs] = \
            weight[self.face_keypoint_idxs] * face_weight

        # 2nd priority: use_shoulder_hip_only
        if use_shoulder_hip_only:
            weight = torch.zeros([n_keypoints]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            if shoulder_weight is not None and hip_weight is not None and \
                    body_weight * face_weight * hand_weight == 0.0:
                weight[self.shoulder_keypoint_idxs] = \
                    weight[self.shoulder_keypoint_idxs] * shoulder_weight
                weight[self.hip_keypoint_idxs] = \
                    weight[self.hip_keypoint_idxs] * hip_weight
            else:
                self.logger.error(
                    'use_shoulder_hip_only is deprecated, '
                    'please manually set: body_weight=0.0, face_weight=0.0, '
                    'hand_weight=0.0, shoulder_weight=1.0, hip_weight=1.0 to '
                    'make sure correct weights are set.')
                raise ValueError

        # 1st priority: keypoints ignored
        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight
