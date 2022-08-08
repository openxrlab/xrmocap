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
from . import campus, human_data, panoptic  # noqa:F401

# yapf: enable
KEYPOINTS_FACTORY['campus'] = campus.CAMPUS_KEYPOINTS
KEYPOINTS_FACTORY['panoptic'] = panoptic.PANOPTIC_KEYPOINTS


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
