# yapf: disable
import mmcv
import numpy as np
import pytest
from os.path import join
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.ops.triangulation.point_selection.builder import (
    build_point_selector,
)

# yapf: enable

TEST_DATA_ROOT = 'test/data/ops/test_triangulation'
CONFIG_ROOT = 'config/ops/triangulation'
CONFIG_TRIANGULATOR = join(CONFIG_ROOT, 'aniposelib_triangulator.py')
CONFIG_CAM_SELECTOR = join(CONFIG_ROOT, 'camera_error_selector.py')
CONFIG_SLOW_CAM_SELECTOR = join(CONFIG_ROOT, 'slow_camera_error_selector.py')


def test_camera_error_selector():
    keypoints2d = np.load(join(TEST_DATA_ROOT,
                               'keypoints2d.npz'))['keypoints2d']
    n_view, _, _, _ = keypoints2d.shape
    cam_param_list = []
    for kinect_index in range(n_view):
        cam_param_path = join(TEST_DATA_ROOT, f'cam_{kinect_index:02d}.json')
        cam_param = FisheyeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    # build a triangulator
    triangulator_config = dict(mmcv.Config.fromfile(CONFIG_TRIANGULATOR))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # build a camera selector
    camera_selector = dict(mmcv.Config.fromfile(CONFIG_CAM_SELECTOR))
    camera_selector['triangulator']['camera_parameters'] = \
        triangulator.camera_parameters
    camera_selector['target_camera_number'] = \
        len(triangulator.camera_parameters) - 1
    camera_selector = build_point_selector(camera_selector)
    # test camera indexes
    camera_indexes = camera_selector.get_camera_indexes(points=keypoints2d)
    assert len(camera_indexes) == len(triangulator.camera_parameters) - 1
    # test camera mask
    init_mask = np.ones_like(keypoints2d[..., 0:1])
    keypoints2d_backup = keypoints2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = camera_selector.get_selection_mask(keypoints2d, init_mask)
    assert np.all(keypoints2d_backup == keypoints2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    assert np.all(points2d_mask.shape == init_mask.shape)
    # test some view has been masked
    init_mask = np.ones_like(keypoints2d[..., 0:1])
    init_mask[0, ...] = 0
    camera_indexes = camera_selector.get_camera_indexes(keypoints2d, init_mask)
    assert len(camera_indexes) == len(triangulator.camera_parameters) - 1
    points2d_mask = camera_selector.get_selection_mask(keypoints2d, init_mask)
    assert np.allclose(points2d_mask, init_mask, equal_nan=True)
    assert np.all(points2d_mask.shape == init_mask.shape)
    init_mask[0:2, ...] = 0
    # expect n-1 views, but only n-2 views are valid
    with pytest.raises(ValueError):
        camera_selector.get_selection_mask(keypoints2d, init_mask)


def test_slow_camera_error_selector():
    keypoints2d = np.load(join(TEST_DATA_ROOT,
                               'keypoints2d.npz'))['keypoints2d']
    n_view, _, _, _ = keypoints2d.shape
    cam_param_list = []
    for kinect_index in range(n_view):
        cam_param_path = join(TEST_DATA_ROOT, f'cam_{kinect_index:02d}.json')
        cam_param = FisheyeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    # build a triangulator
    triangulator_config = dict(mmcv.Config.fromfile(CONFIG_TRIANGULATOR))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # build a camera selector
    camera_selector = dict(mmcv.Config.fromfile(CONFIG_SLOW_CAM_SELECTOR))
    camera_selector['triangulator']['camera_parameters'] = \
        triangulator.camera_parameters
    camera_selector['target_camera_number'] = \
        len(triangulator.camera_parameters) - 1
    camera_selector = build_point_selector(camera_selector)
    # test camera indexes
    camera_indexes = camera_selector.get_camera_indexes(points=keypoints2d)
    assert len(camera_indexes) == len(triangulator.camera_parameters) - 1
    # test camera mask
    init_mask = np.ones_like(keypoints2d[..., 0:1])
    keypoints2d_backup = keypoints2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = camera_selector.get_selection_mask(keypoints2d, init_mask)
    assert np.all(keypoints2d_backup == keypoints2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    assert np.all(points2d_mask.shape == init_mask.shape)