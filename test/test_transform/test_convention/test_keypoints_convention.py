# yapf: disable
import numpy as np
import torch

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_keypoint_num, get_mapping_dict,
)

# yapf: enable


def test_get_mapping_dict():
    map_dict = get_mapping_dict(src='human_data', dst='coco')
    assert len(map_dict) == get_keypoint_num(convention='coco')
    for k, v in map_dict.items():
        assert k >= 0 and k < 190
        assert v >= 0 and v < get_keypoint_num(convention='coco')


def test_convert_keypoints():
    # test convert np
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.ones(shape=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    assert isinstance(kps.get_keypoints(), np.ndarray)
    hd_kps = convert_keypoints(keypoints=kps, dst='human_data')
    assert isinstance(hd_kps.get_keypoints(), np.ndarray)
    single_mask = hd_kps.get_mask()[0, 0]
    assert single_mask.sum() == mask_np.shape[-1]
    # test convert torch
    kps = kps.to_tensor()
    assert isinstance(kps.get_keypoints(), torch.Tensor)
    hd_kps = convert_keypoints(keypoints=kps, dst='human_data')
    assert isinstance(hd_kps.get_keypoints(), torch.Tensor)
    single_mask = hd_kps.get_mask()[0, 0]
    assert single_mask.sum() == mask_np.shape[-1]
