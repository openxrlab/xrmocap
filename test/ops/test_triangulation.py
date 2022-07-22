import mmcv
import numpy as np
import os
import pytest
import shutil
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.utils.triangulation_utils import parse_keypoints_mask

input_dir = 'test/data/ops/test_triangulation'
output_dir = 'test/data/output/ops/test_triangulation'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_aniposelib_triangulator():
    keypoints2d_dict = dict(
        np.load(os.path.join(input_dir, 'keypoints2d.npz')))
    keypoints2d = keypoints2d_dict['keypoints2d']
    view_n, frame_n, keypoint_n, _ = keypoints2d.shape
    keypoints2d_mask = keypoints2d_dict['keypoints2d_mask']
    cam_param_list = []
    for kinect_index in range(view_n):
        cam_param_path = os.path.join(input_dir,
                                      f'cam_{kinect_index:02d}.json')
        cam_param = FisheyeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    triangulator_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/aniposelib_triangulator.py'))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # test kp2d np
    keypoints3d = triangulator.triangulate(keypoints2d)
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test kp2d list
    keypoints3d = triangulator.triangulate(keypoints2d.tolist())
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test kp2d tuple
    keypoints3d = triangulator.triangulate(tuple(map(tuple, keypoints2d)))
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test mask np
    points_mask = np.ones_like(keypoints2d[..., 0:1])
    keypoints3d = triangulator.triangulate(
        points=keypoints2d, points_mask=points_mask)
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test mask list
    keypoints3d = triangulator.triangulate(
        points=keypoints2d, points_mask=points_mask.tolist())
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test mask tuple
    keypoints3d = triangulator.triangulate(
        points=keypoints2d, points_mask=tuple(map(tuple, points_mask)))
    # test mask from confidence
    points_mask = parse_keypoints_mask(
        keypoints=keypoints2d, keypoints_mask=keypoints2d_mask)
    keypoints3d = triangulator.triangulate(
        points=keypoints2d, points_mask=points_mask)
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test no confidence
    keypoints3d = triangulator.triangulate(points=keypoints2d[..., :2])
    assert keypoints3d.shape[:2] == keypoints2d.shape[1:3]
    # test wrong type
    with pytest.raises(TypeError):
        triangulator.triangulate(points='points')
    with pytest.raises(TypeError):
        triangulator.triangulate(points=keypoints2d, points_mask='points_mask')
    # test wrong shape
    with pytest.raises(ValueError):
        triangulator.triangulate(points=keypoints2d[:2, ...])
    with pytest.raises(ValueError):
        triangulator.triangulate(points=keypoints2d[:1, ...])
    with pytest.raises(ValueError):
        triangulator.triangulate(
            points=keypoints2d, points_mask=points_mask[:2, ...])
    # test slice
    int_triangulator = triangulator[0]
    assert len(int_triangulator.camera_parameters) == 1
    list_triangulator = triangulator[[0, 1]]
    assert len(list_triangulator.camera_parameters) == 2
    tuple_triangulator = triangulator[(0, 1)]
    assert len(tuple_triangulator.camera_parameters) == 2
    slice_triangulator = triangulator[:2]
    assert len(slice_triangulator.camera_parameters) == 2
    slice_triangulator = triangulator[::2]
    assert len(slice_triangulator.camera_parameters) >= 2
    # test error
    keypoints3d = triangulator.triangulate(keypoints2d)
    error = triangulator.get_reprojection_error(
        points2d=keypoints2d, points3d=keypoints3d)
    assert np.all(error.shape == keypoints2d[..., :2].shape)
