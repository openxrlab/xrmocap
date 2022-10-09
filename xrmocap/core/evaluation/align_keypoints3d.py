# yapf: disable
import numpy as np
import string
from typing import List

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_idx,
)
from xrmocap.transform.limbs import get_limbs_from_keypoints
from xrmocap.utils.mvpose_utils import (
    add_campus_jaw_headtop, add_campus_jaw_headtop_mask,
)

# yapf: enable


def align_keypoints3d(pred_keypoints3d: Keypoints, gt_keypoints3d: Keypoints,
                      eval_kps3d_convention: string,
                      selected_limbs_name: List[List[str]],
                      additional_limbs_names: List[List[str]]):
    """align keypoints convention.

    Args:
        pred_keypoints3d (Keypoints): prediction of keypoints
        gt_keypoints3d (Keypoints): ground true of keypoints
        eval_kps3d_convention (string): keypoints convention to align
        selected_limbs_name (List): selected limbs to be evaluated
        additional_limbs_names (List): additional limbs to be evaluated
    """
    ret_limbs = []
    gt_nose = None
    pred_nose = None
    pred_kps3d_convention = pred_keypoints3d.get_convention()
    gt_kps3d_convention = gt_keypoints3d.get_convention()
    if gt_kps3d_convention == 'panoptic':
        gt_nose_index = get_keypoint_idx(
            name='nose_openpose', convention=gt_kps3d_convention)
        gt_nose = gt_keypoints3d.get_keypoints()[:, :, gt_nose_index, :3]

    if pred_kps3d_convention == 'coco':
        pred_nose_index = get_keypoint_idx(
            name='nose', convention=pred_kps3d_convention)
        pred_nose = pred_keypoints3d.get_keypoints()[:, :, pred_nose_index, :3]

    if pred_kps3d_convention == 'fourdag_19' or\
            pred_kps3d_convention == 'openpose_25':
        pred_leftear_index = get_keypoint_idx(
            name='left_ear_openpose', convention=pred_kps3d_convention)
        pre_rightear_index = get_keypoint_idx(
            name='right_ear_openpose', convention=pred_kps3d_convention)
        head_center = (
            pred_keypoints3d.get_keypoints()[:, :, pred_leftear_index, :3] +
            pred_keypoints3d.get_keypoints()[:, :, pre_rightear_index, :3]) / 2
        pred_nose = head_center

    if pred_kps3d_convention != eval_kps3d_convention:
        pred_keypoints3d = convert_keypoints(
            keypoints=pred_keypoints3d,
            dst=eval_kps3d_convention,
            approximate=True)
    if gt_kps3d_convention != eval_kps3d_convention:
        gt_keypoints3d = convert_keypoints(
            keypoints=gt_keypoints3d,
            dst=eval_kps3d_convention,
            approximate=True)
    limbs = get_limbs_from_keypoints(
        keypoints=pred_keypoints3d, fill_limb_names=True)
    limb_name_list = []
    conn_list = []
    for limb_name, conn in limbs.get_connections_by_names().items():
        limb_name_list.append(limb_name)
        conn_list.append(conn)

    for idx, limb_name in enumerate(limb_name_list):
        if limb_name in selected_limbs_name:
            ret_limbs.append(conn_list[idx])

    for conn_names in additional_limbs_names:
        kps_idx_0 = get_keypoint_idx(
            name=conn_names[0], convention=eval_kps3d_convention)
        kps_idx_1 = get_keypoint_idx(
            name=conn_names[1], convention=eval_kps3d_convention)
        ret_limbs.append(np.array([kps_idx_0, kps_idx_1], dtype=np.int32))
    pred_kps3d_mask = pred_keypoints3d.get_mask()
    pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
    if pred_nose is not None:
        pred_kps3d = add_campus_jaw_headtop(pred_nose, pred_kps3d)
        pred_kps3d_mask = add_campus_jaw_headtop_mask(pred_kps3d_mask)

    gt_kps3d_mask = gt_keypoints3d.get_mask()
    gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
    if gt_nose is not None:
        gt_kps3d = add_campus_jaw_headtop(gt_nose, gt_kps3d)
        gt_kps3d_mask = add_campus_jaw_headtop_mask(gt_kps3d_mask)

    pred_kps3d = np.concatenate((pred_kps3d, pred_kps3d_mask[..., np.newaxis]),
                                axis=-1)
    pred_keypoints3d = Keypoints(
        kps=pred_kps3d, mask=pred_kps3d_mask, convention=eval_kps3d_convention)
    gt_kps3d = np.concatenate((gt_kps3d, gt_kps3d_mask[..., np.newaxis]),
                              axis=-1)
    gt_keypoints3d = Keypoints(
        kps=gt_kps3d, mask=gt_kps3d_mask, convention=eval_kps3d_convention)

    return pred_keypoints3d, gt_keypoints3d, ret_limbs
