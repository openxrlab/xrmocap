import os

import mmcv
import numpy as np
from xrprimer.data_structure.camera.pinhole_camera import \
    PinholeCameraParameter  # noqa:E501

from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.ops.triangulation.point_selection.builder import \
    build_point_selector  # prevent linting conflicts


def test_camera_error_selector():
    keypoints2d = np.load(
        os.path.join('test/data/test_ops/test_triangulation',
                     'keypoints2d.npz'))['arr_0']
    view_n, frame_n, keypoint_n, _ = keypoints2d.shape
    cam_param_list = []
    for kinect_index in range(view_n):
        cam_param_path = os.path.join('test/data/test_ops/test_triangulation',
                                      f'cam_{kinect_index:02d}.json')
        cam_param = PinholeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    # build a triangulator
    triangulator_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/aniposelib_triangulator.py'))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # build a camera selector
    camera_selector = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/camera_error_selector.py'))
    camera_selector['triangulator']['camera_parameters'] = \
        triangulator.camera_parameters
    camera_selector['target_camera_number'] = \
        len(triangulator.camera_parameters) - 1
    camera_selector = build_point_selector(camera_selector)
    # test camera indices
    camera_indices = camera_selector.get_camera_indices(points=keypoints2d)
    assert len(camera_indices) == len(triangulator.camera_parameters) - 1
    # test camera mask
    init_mask = np.ones_like(keypoints2d[..., 0:1])
    points2d_backup = keypoints2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = camera_selector.get_selection_mask(keypoints2d, init_mask)
    assert np.all(points2d_backup == keypoints2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    assert np.all(points2d_mask.shape == init_mask.shape)


def test_slow_camera_error_selector():
    keypoints2d = np.load(
        os.path.join('test/data/test_ops/test_triangulation',
                     'keypoints2d.npz'))['arr_0']
    view_n, frame_n, keypoint_n, _ = keypoints2d.shape
    cam_param_list = []
    for kinect_index in range(view_n):
        cam_param_path = os.path.join('test/data/test_ops/test_triangulation',
                                      f'cam_{kinect_index:02d}.json')
        cam_param = PinholeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    # build a triangulator
    triangulator_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/aniposelib_triangulator.py'))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # build a camera selector
    camera_selector = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/slow_camera_error_selector.py'))
    camera_selector['triangulator']['camera_parameters'] = \
        triangulator.camera_parameters
    camera_selector['target_camera_number'] = \
        len(triangulator.camera_parameters) - 1
    camera_selector = build_point_selector(camera_selector)
    # test camera indices
    camera_indices = camera_selector.get_camera_indices(points=keypoints2d)
    assert len(camera_indices) == len(triangulator.camera_parameters) - 1
    # test camera mask
    init_mask = np.ones_like(keypoints2d[..., 0:1])
    points2d_backup = keypoints2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = camera_selector.get_selection_mask(keypoints2d, init_mask)
    assert np.all(points2d_backup == keypoints2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    assert np.all(points2d_mask.shape == init_mask.shape)
