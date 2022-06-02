# yapf: disable
import numpy as np
from typing import Tuple, Union

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.data_structure.limbs import Limbs
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoints_factory, get_mapping_dict, human_data,
)

# yapf: enable


def get_limbs_from_keypoints(
    kps: Keypoints,
    frame_idx: Union[int, None] = None,
    person_idx: Union[int, None] = None,
    keypoints_factory: Union[dict, None] = None,
) -> Limbs:
    """Get an instance of class Limbs, from a Keypoints instance. It searches
    existing limbs in HumanData convention.

    Args:
        keypoints (Keypoints):
            An instance of Keypoints class.
        frame_idx (Union[int, None], optional):
            If given, limbs.points will be
            set to keypoints[frame_idx, person_idx, :, :].
            Defaults to None, limbs.points will not be set.
        person_idx (Union[int, None], optional):
            If given, limbs.points will be
            set to keypoints[frame_idx, person_idx, :, :].
            Defaults to None, limbs.points will not be set.
        keypoints_factory (Union[dict, None], optional):
            A dict to store all the keypoint conventions.
            Defaults to None, KEYPOINTS_FACTORY will be set.

    Returns:
        Limbs: an instance of class Limbs.
    """
    logger = kps.logger
    if keypoints_factory is None:
        keypoints_factory = get_keypoints_factory()
    limbs_source = human_data.HUMAN_DATA_LIMBS_INDEX
    kps_np = kps.to_numpy()
    # if both frame_idx and person_idx are set
    # take the frame and person
    if frame_idx is not None and person_idx is not None:
        frame_idx = int(frame_idx)
        person_idx = int(person_idx)
        selected_mask = kps.get_mask()[frame_idx, person_idx, :]
        selected_kps = kps_np.get_keypoints()[frame_idx, person_idx, ...]
    # if any of [frame_idx, person_idx] not configured
    # use or result of masks
    else:
        if frame_idx is not None or person_idx is not None:
            logger.warning('Either frame_idx or person_idx has not' +
                           ' been set properly, limbs.points will not be set.')
        kps_n = kps_np.get_keypoints_number()
        flat_mask = kps_np.get_mask().reshape(-1, kps_n)
        or_mask = np.sign(np.sum(flat_mask, axis=0))
        selected_mask = or_mask
        selected_kps = kps_np.get_keypoints()[0, 0, ...]
    one_frame_kps = Keypoints(
        dtype='numpy',
        kps=selected_kps,
        mask=selected_mask,
        convention=kps_np.get_convention(),
        logger=kps.logger)
    human_data_kps = convert_keypoints(
        keypoints=one_frame_kps,
        dst='human_data',
        keypoints_factory=keypoints_factory)
    mapping_back = get_mapping_dict(
        src='human_data',
        dst=kps.get_convention(),
        keypoints_factory=keypoints_factory)
    mask = human_data_kps.get_mask()[0, 0, :]
    connecntions = []
    parts = []
    part_names = []
    for part_name, part_limbs in limbs_source.items():
        part_record = []
        for limb in part_limbs:
            # points index defined in human data
            hd_start_idx = limb[0]
            hd_end_idx = limb[1]
            # both points exits
            if mask[hd_start_idx] + mask[hd_end_idx] == 2:
                src_start_idx = mapping_back[hd_start_idx]
                src_end_idx = mapping_back[hd_end_idx]
                connecntions.append([src_start_idx, src_end_idx])
                part_record.append(len(connecntions) - 1)
        if len(part_record) > 0:
            parts.append(part_record)
            part_names.append(part_name)
    points = None if person_idx is None or frame_idx is None else\
        one_frame_kps.get_keypoints()[frame_idx, person_idx, ...]
    ret_limbs = Limbs(
        connections=np.asarray(connecntions),
        parts=parts,
        part_names=part_names,
        points=points,
        logger=kps.logger)
    return ret_limbs


def search_limbs(data_source: str,
                 mask: Union[np.ndarray, tuple, list] = None,
                 keypoints_factory: dict = None) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis human_data limbs.

    The
    mask could mask out the incorrect keypoints.
    Args:
        data_source (str): data source type.
        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to keypoints_mapping. Defaults to None.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    if keypoints_factory is None:
        keypoints_factory = get_keypoints_factory()
    limbs_source = human_data.HUMAN_DATA_LIMBS_INDEX
    limbs_palette = human_data.HUMAN_DATA_PALETTE
    keypoints_source = keypoints_factory['human_data']
    keypoints_target = keypoints_factory[data_source]
    limbs_target = {}
    for k, part_limbs in limbs_source.items():
        limbs_target[k] = []
        for limb in part_limbs:
            flag = False
            if (keypoints_source[limb[0]]
                    in keypoints_target) and (keypoints_source[limb[1]]
                                              in keypoints_target):
                if mask is not None:
                    if mask[keypoints_target.index(keypoints_source[
                            limb[0]])] != 0 and mask[keypoints_target.index(
                                keypoints_source[limb[1]])] != 0:
                        flag = True
                else:
                    flag = True
                if flag:
                    limbs_target.setdefault(k, []).append([
                        keypoints_target.index(keypoints_source[limb[0]]),
                        keypoints_target.index(keypoints_source[limb[1]])
                    ])
        if k in limbs_target:
            if k == 'body':
                np.random.seed(0)
                limbs_palette[k] = np.random.randint(
                    0, high=255, size=(len(limbs_target[k]), 3))
            else:
                limbs_palette[k] = np.array(limbs_palette[k])
    return limbs_target, limbs_palette
