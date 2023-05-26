# yapf: disable
import numpy as np
from typing import List, Union
from xrprimer.data_structure import Keypoints
from xrprimer.transform.convention.keypoints_convention import (
    KEYPOINTS_FACTORY,
)
from xrprimer.transform.convention.keypoints_convention import \
    convert_keypoints as convert_keypoints_xrprimer
from xrprimer.transform.convention.keypoints_convention import (
    get_keypoint_num, get_mapping,
)
from xrprimer.utils.log_utils import get_logger, logging

from . import fourdag_19, human_data, panoptic
from .paf import ALL_PAF_MAPPING

try:
    from mmhuman3d.core.conventions.keypoints_mapping import \
        KEYPOINTS_FACTORY as KEYPOINTS_FACTORY_MM
    has_mmhuman3d = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmhuman3d = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable
if isinstance(KEYPOINTS_FACTORY, dict):
    KEYPOINTS_FACTORY['panoptic_15'] = panoptic.PANOPTIC15_KEYPOINTS
    KEYPOINTS_FACTORY['fourdag_19'] = fourdag_19.FOURDAG19_KEYPOINTS
if has_mmhuman3d:
    KEYPOINTS_FACTORY_MM.update(KEYPOINTS_FACTORY)

convert_keypoints_warned = False


def convert_keypoints(
        keypoints: Keypoints,
        dst: str,
        approximate: bool = False,
        keypoints_factory: dict = KEYPOINTS_FACTORY,
        logger: Union[None, str, logging.Logger] = None) -> Keypoints:
    global convert_keypoints_warned
    if not convert_keypoints_warned:
        convert_keypoints_warned = True
        logger = get_logger(logger)
        logger.warning(
            'Keypoints defined in XRMoCap is deprecated,' +
            ' use `from xrprimer.data_structure import Keypoints` instead.' +
            ' This class will be removed from XRMoCap before v0.9.0.')
    return convert_keypoints_xrprimer(
        keypoints=keypoints,
        dst=dst,
        approximate=approximate,
        keypoints_factory=keypoints_factory,
        logger=logger)


def get_keypoint_idxs_by_part(
        part: str,
        convention: str = 'smplx',
        human_data_parts: dict = human_data.HUMAN_DATA_PARTS,
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[int]:
    """Get part keypoints indices from specified part and convention.

    Args:
        part (str): part to search from
        convention (str): data type from keypoints_factory.
            Defaults to 'smplx'.
        human_data_parts (dict, optional): A dict to store the part
            keypoints. Defaults to human_data.HUMAN_DATA_PARTS.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        List[int]: part keypoint indices
    """
    keypoints = keypoints_factory[convention]
    if part not in human_data_parts.keys():
        raise ValueError('part not in allowed parts')
    part_keypoints = list(set(human_data_parts[part]) & set(keypoints))
    part_keypoints_idx = [keypoints.index(kp) for kp in part_keypoints]
    return part_keypoints_idx


def convert_bottom_up_kps_paf(
    kps_paf: List,
    src: str,
    dst: str,
    approximate: bool = False,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
):
    """Convert keypoints and pafs following the mapping correspondence between
    src and dst keypoints definition.

    Args:
        kps_paf (List):
            A list of dict of 2D keypoints and pafs in shape
                 [{'kps':[],'pafs':[]},...]
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
        dst_detections (list): the destination keypoints and paf
    """
    n_frame = len(kps_paf)
    dst_n_kps = get_keypoint_num(
        convention=dst, keypoints_factory=keypoints_factory)
    dst_idxs, src_idxs, _ = \
        get_mapping(src, dst, approximate, keypoints_factory)
    paf_mapping = ALL_PAF_MAPPING[src][dst]

    dst_detections = []
    for i in range(n_frame):
        var = {
            'kps': [np.array([]) for j in range(dst_n_kps)],
            'pafs': [np.array([]) for k in range(len(paf_mapping))]
        }
        dst_detections.append(var)
    for frame_id in range(n_frame):
        for i in range(len(dst_idxs)):
            dst_detections[frame_id]['kps'][dst_idxs[i]] = np.array(
                kps_paf[frame_id]['kps'][src_idxs[i]], dtype=np.float32)
        for i in range(len(paf_mapping)):
            if isinstance(paf_mapping[i], list):
                if paf_mapping[i][0] < 0:
                    dst_detections[frame_id]['pafs'][i] = np.array(
                        kps_paf[frame_id]['pafs'][-paf_mapping[i][0]],
                        dtype=np.float32).T
                else:
                    dst_detections[frame_id]['pafs'][i] = np.array(
                        kps_paf[frame_id]['pafs'][paf_mapping[i][0]],
                        dtype=np.float32)
                dst_detections[frame_id]['pafs'][
                    i] = dst_detections[frame_id]['pafs'][i] * (
                        dst_detections[frame_id]['pafs'][i] > 0.1)
                for path_id in paf_mapping[i][1:]:
                    if path_id < 0:
                        arr = np.array(
                            kps_paf[frame_id]['pafs'][-path_id],
                            dtype=np.float32).T
                    else:
                        arr = np.array(
                            kps_paf[frame_id]['pafs'][path_id],
                            dtype=np.float32)
                    dst_detections[frame_id]['pafs'][i] = np.matmul(
                        dst_detections[frame_id]['pafs'][i], arr)
                    dst_detections[frame_id]['pafs'][
                        i] = dst_detections[frame_id]['pafs'][i] * (
                            dst_detections[frame_id]['pafs'][i] > 0.1)
                dst_detections[frame_id]['pafs'][i] = dst_detections[frame_id][
                    'pafs'][i] * len(paf_mapping[i])
            else:
                if paf_mapping[i] < 0:
                    dst_detections[frame_id]['pafs'][i] = np.array(
                        kps_paf[frame_id]['pafs'][-paf_mapping[i]],
                        dtype=np.float32).T
                else:
                    dst_detections[frame_id]['pafs'][i] = np.array(
                        kps_paf[frame_id]['pafs'][paf_mapping[i]],
                        dtype=np.float32)
    return dst_detections


def get_intersection_mask(convention_a: str,
                          convention_b: str,
                          dst_convention: str = 'human_data') -> np.ndarray:
    """Get a intersection mask of two different conventions.

    Args:
        convention_a (str):
            First convention name.
        convention_b (str):
            Second convention name.
        dst_convention (str):
            Destination convention name.
            Defaults to human_data.

    Returns:
        np.ndarray:
            Intersection mask in the format of
            destination convention. [1, 1, n_kps]
    """

    def get_converted_mask(src_convention, dst_convention):
        n_kps = get_keypoint_num(src_convention)
        dummy_kps3d = np.ones((1, 1, n_kps, 4))
        dummy_keypoints3d = Keypoints(
            dtype='numpy',
            kps=dummy_kps3d,
            mask=dummy_kps3d[..., -1] > 0,
            convention=src_convention)

        dummy_keypoints3d_converted = convert_keypoints(
            dummy_keypoints3d, dst=dst_convention, approximate=True)
        converted_mask = dummy_keypoints3d_converted.get_mask()
        return converted_mask

    converted_mask_a = get_converted_mask(convention_a, dst_convention)
    converted_mask_b = get_converted_mask(convention_b, dst_convention)
    intersection_mask = np.multiply(converted_mask_a, converted_mask_b)

    return intersection_mask
