# yapf: disable
import numpy as np

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.keypoints3d.optim.builder import (
    build_keypoints3d_optimizer,
)

# yapf: enable


def test_nan_interpolation():
    # 3 frames, 2 people, 4 kps
    kps_arr = np.zeros(shape=[3, 2, 5, 4])
    mask = np.ones_like(kps_arr[..., 0])
    kps_arr[1, 0, :, :] = np.nan
    kps_arr[2, 0, :, :] = 20
    keypoints3d = Keypoints(
        dtype='numpy', kps=kps_arr, mask=mask, convention='non_exist_conv')
    kps_arr_backup = kps_arr.copy()
    cfg = dict(type='NanInterpolation')
    optim = build_keypoints3d_optimizer(cfg)
    # test numpy keypoints3d
    optimed_keypoints3d = optim.optimize_keypoints3d(keypoints3d)
    # assert input not changed
    assert np.isnan(keypoints3d.get_keypoints()[1, 0, 0, 0])
    assert np.all(keypoints3d.get_keypoints()[2, ...] == kps_arr_backup[2,
                                                                        ...])
    # the second person should be the same
    assert np.all(optimed_keypoints3d.get_keypoints()[:, 1, :, :] ==
                  keypoints3d.get_keypoints()[:, 1, :, :])
    # the first person has been interpolated
    assert optimed_keypoints3d.get_keypoints()[1, 0, 0, 0] == 10
    assert not np.any(np.isnan(optimed_keypoints3d.get_keypoints()))
    # test torch keypoints3d
    optimed_keypoints3d = optim.optimize_keypoints3d(keypoints3d.to_tensor())
    # the second person should be the same
    assert np.all(optimed_keypoints3d.get_keypoints()[:, 1, :, :] ==
                  keypoints3d.get_keypoints()[:, 1, :, :])
    # the first person has been interpolated
    assert optimed_keypoints3d.get_keypoints()[1, 0, 0, 0] == 10
    assert not np.any(np.isnan(optimed_keypoints3d.get_keypoints()))
