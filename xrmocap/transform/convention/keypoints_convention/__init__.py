# yapf: disable
import numpy as np
import torch
from mmhuman3d.core.conventions.keypoints_mapping import (  # noqa:F401
    KEYPOINTS_FACTORY,
)
from mmhuman3d.core.conventions.keypoints_mapping import \
    convert_kps as convert_kps_mm  # noqa:F401
from mmhuman3d.core.conventions.keypoints_mapping import (  # noqa:F401
    get_keypoint_idx, get_keypoint_idxs_by_part, get_keypoint_num, get_mapping,
)
from typing import List

from xrmocap.data_structure.keypoints import Keypoints
from . import campus, human_data, panoptic, fourdag19  # noqa:F401

# yapf: enable
KEYPOINTS_FACTORY['campus'] = campus.CAMPUS_KEYPOINTS
KEYPOINTS_FACTORY['panoptic'] = panoptic.PANOPTIC_KEYPOINTS
KEYPOINTS_FACTORY['fourdag19'] = fourdag19.FOURDAG19_KEYPOINTS


def convert_keypoints(
    keypoints: Keypoints,
    dst: str,
    approximate: bool = False,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Keypoints:
    """Convert keypoints following the mapping correspondence between src and
    dst keypoints definition.

    Args:
        keypoints (Keypoints):
            An instance of Keypoints class.
        dst (str):
            The name of destination convention.
        approximate (bool, optional):
            Whether approximate mapping is allowed.
            Defaults to False.
        keypoints_factory (dict, optional):
            A dict to store all the keypoint conventions.
            Defaults to KEYPOINTS_FACTORY.

    Returns:
        Keypoints:
            An instance of Keypoints class, whose convention is dst,
            and dtype, device are same as input.
    """
    src = keypoints.get_convention()
    src_arr = keypoints.get_keypoints()
    n_frame, n_person, kps_n, dim = src_arr.shape
    flat_arr = src_arr.reshape(-1, kps_n, dim)
    flat_mask = keypoints.get_mask().reshape(-1, kps_n)

    if isinstance(src_arr, torch.Tensor):

        def new_array_func(shape, value, ref_data, if_uint8):
            if if_uint8:
                dtype = torch.uint8
            else:
                dtype = ref_data.dtype
            if value == 1:
                return torch.ones(
                    size=shape, dtype=dtype, device=ref_data.device)
            elif value == 0:
                return torch.zeros(
                    size=shape, dtype=dtype, device=ref_data.device)
            else:
                raise ValueError

    elif isinstance(src_arr, np.ndarray):

        def new_array_func(shape, value, ref_data, if_uint8):
            if if_uint8:
                dtype = np.uint8
            else:
                dtype = ref_data.dtype
            if value == 1:
                return np.ones(shape=shape)
            elif value == 0:
                return np.zeros(shape=shape, dtype=dtype)
            else:
                raise ValueError

    dst_n_kps = get_keypoint_num(
        convention=dst, keypoints_factory=keypoints_factory)
    dst_idxs, src_idxs, _ = \
        get_mapping(src, dst, approximate, keypoints_factory)
    # multi frame multi person kps
    dst_arr = new_array_func(
        shape=(n_frame * n_person, dst_n_kps, dim),
        value=0,
        ref_data=src_arr,
        if_uint8=False)
    # multi frame multi person mask
    dst_mask = new_array_func(
        shape=(n_frame * n_person, dst_n_kps),
        value=0,
        ref_data=src_arr,
        if_uint8=True)
    # mapping from source
    dst_mask[:, dst_idxs] = flat_mask[:, src_idxs]
    dst_arr[:, dst_idxs, :] = flat_arr[:, src_idxs, :]
    multi_mask = dst_mask.reshape(n_frame, n_person, dst_n_kps)
    multi_arr = dst_arr.reshape(n_frame, n_person, dst_n_kps, dim)
    ret_kps = Keypoints(
        dtype=keypoints.dtype,
        kps=multi_arr,
        mask=multi_mask,
        convention=dst,
        logger=keypoints.logger)
    return ret_kps

def convert_bottom_up_kps_paf(
    kps_paf: List,
    src: str,
    dst: str,
    approximate: bool = False,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Keypoints:
    """Convert keypoints following the mapping correspondence between src and
    dst keypoints definition.

    Args:
        keypoints (Keypoints):
            An instance of Keypoints class.
        dst (str):
            The name of destination convention.
        approximate (bool, optional):
            Whether approximate mapping is allowed.
            Defaults to False.
        keypoints_factory (dict, optional):
            A dict to store all the keypoint conventions.
            Defaults to KEYPOINTS_FACTORY.

    Returns:
        Keypoints:
            An instance of Keypoints class, whose convention is dst,
            and dtype, device are same as input.
    """
    # for joint_id in range(n_kps):
    #     tmp_joint_id = joint_mapping[joint_id]
    #     if tmp_joint_id != -1:
    #         convert_detections[frame_id]['joints'][tmp_joint_id] = np.array(detections[frame_id]['joints'][joint_id], dtype=np.float32)
    # for paf_id in range(n_pafs):
    #     tmp_paf_id = pafs_mapping[paf_id]
    #     if tmp_paf_id != -1: 
    #         convert_detections[frame_id]['pafs'][tmp_paf_id]= np.array(detections[frame_id]['pafs'][paf_id], dtype=np.float32)

    n_frame = len(kps_paf)
    dst_n_kps = get_keypoint_num(
        convention=dst, keypoints_factory=keypoints_factory)
    dst_detections = []
    for i in range(n_frame):
        var = {'joints':[np.array([]) for j in range(dst_n_kps)], 'pafs':[np.array([]) for k in range(dst_n_kps)]}
        dst_detections.append(var)
    dst_idxs, src_idxs, _ = \
        get_mapping(src, dst, approximate, keypoints_factory)
    pafs_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, 16, -1, -1, 17, -1, -1]
    # joint_mapping = [4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, -1, -1, 18, -1, -1]

    for frame_id in range(n_frame):
        for  i in range(len(dst_idxs)):
            dst_detections[frame_id]['joints'][dst_idxs[i]] = np.array(kps_paf[frame_id]['joints'][src_idxs[i]], dtype=np.float32)

        for paf_id in range(len(pafs_mapping)):
            tmp_paf_id = pafs_mapping[paf_id]
            if tmp_paf_id != -1: 
                dst_detections[frame_id]['pafs'][tmp_paf_id]= np.array(kps_paf[frame_id]['pafs'][paf_id], dtype=np.float32)
    return dst_detections
    
def get_keypoints_factory() -> dict:
    """Get the KEYPOINTS_FACTORY defined in keypoints convention.

    Returns:
        dict:
            KEYPOINTS_FACTORY whose keys are convention
            names and values are keypoints lists.
    """
    return KEYPOINTS_FACTORY


def get_mapping_dict(src: str,
                     dst: str,
                     approximate: bool = False,
                     keypoints_factory: dict = KEYPOINTS_FACTORY) -> dict:
    """Call get_mapping from mmhuman3d and make a dict mapping src index to dst
    index.

    Args:
        src (str):
            The name of source convention.
        dst (str):
            The name of destination convention.
        approximate (bool, optional):
            Whether approximate mapping is allowed.
            Defaults to False.
        keypoints_factory (dict, optional):
            A dict to store all the keypoint conventions.
            Defaults to KEYPOINTS_FACTORY.

    Returns:
        dict:
            A mapping dict whose keys are src indexes
            and values are dst indexes.
    """
    mapping_back = get_mapping(
        src=src,
        dst=dst,
        keypoints_factory=keypoints_factory,
        approximate=approximate)
    inter_to_dst, inter_to_src = mapping_back[:2]
    mapping_dict = {}
    for index in range(len(inter_to_dst)):
        mapping_dict[inter_to_src[index]] = inter_to_dst[index]
    return mapping_dict


def get_keypoint_names(
        convention: str = 'smplx',
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[str]:
    """Get names of keypoints of specified convention.

    Args:
        convention (str): data type from keypoints_factory.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        List[str]: keypoint names
    """
    keypoints = keypoints_factory[convention]
    return keypoints
