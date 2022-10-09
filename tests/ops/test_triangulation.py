import cv2
import mmcv
import numpy as np
import os
import pytest
import shutil
from xrprimer.data_structure.camera import PinholeCameraParameter  # noqa:E501

from xrmocap.ops.triangulation.builder import build_triangulator

input_dir = 'tests/data/ops/test_triangulation'
output_dir = 'tests/data/output/ops/test_triangulation'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_aniposelib_triangulator():
    n_view = 6
    kps2d_list = []
    mask_list = []
    for view_idx in range(n_view):
        npz_path = os.path.join(input_dir, f'keypoints_2d_{view_idx:02d}.npz')
        npz_dict = dict(np.load(npz_path, allow_pickle=True))
        kps2d_list.append(npz_dict['keypoints'][0, 0, :, :])
        mask_list.append(npz_dict['mask'][0, 0, :])
    kps2d = np.asarray(kps2d_list)
    kps2d_mask = np.asarray(mask_list, dtype=kps2d.dtype)
    cam_param_list = []
    for view_idx in range(n_view):
        cam_param_path = os.path.join(input_dir, f'cam_{view_idx:03d}.json')
        cam_param = PinholeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    triangulator_config = dict(
        mmcv.Config.fromfile(
            'configs/modules/ops/triangulation/aniposelib_triangulator.py'))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    # test kp2d np
    kps3d = triangulator.triangulate(kps2d)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test kp2d list
    kps3d = triangulator.triangulate(kps2d.tolist())
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test kp2d tuple
    kps3d = triangulator.triangulate(tuple(map(tuple, kps2d)))
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask np
    points_mask = np.ones_like(kps2d[..., 0:1])
    kps3d = triangulator.triangulate(points=kps2d, points_mask=points_mask)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask list
    kps3d = triangulator.triangulate(
        points=kps2d, points_mask=points_mask.tolist())
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask tuple
    kps3d = triangulator.triangulate(
        points=kps2d, points_mask=tuple(map(tuple, points_mask)))
    # test mask from confidence
    points_mask = kps2d_mask
    kps3d = triangulator.triangulate(points=kps2d, points_mask=points_mask)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test no confidence
    kps3d = triangulator.triangulate(points=kps2d[..., :2])
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test wrong type
    with pytest.raises(TypeError):
        triangulator.triangulate(points='points')
    with pytest.raises(TypeError):
        triangulator.triangulate(points=kps2d, points_mask='points_mask')
    # test wrong shape
    with pytest.raises(ValueError):
        triangulator.triangulate(points=kps2d[:2, ...])
    with pytest.raises(ValueError):
        triangulator.triangulate(points=kps2d[:1, ...])
    with pytest.raises(ValueError):
        triangulator.triangulate(
            points=kps2d, points_mask=points_mask[:2, ...])
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
    kps3d = triangulator.triangulate(kps2d)
    error = triangulator.get_reprojection_error(points2d=kps2d, points3d=kps3d)
    assert np.all(error.shape == kps2d[..., :2].shape)
    error = triangulator.get_reprojection_error(
        points2d=kps2d, points3d=kps3d, reduction='mean')
    assert float(error) != 0
    error = triangulator.get_reprojection_error(
        points2d=kps2d, points3d=kps3d, reduction='sum')
    assert float(error) != 0
    # triangulate and visualize
    keypoints3d = triangulator.triangulate(
        points=kps2d, points_mask=np.expand_dims(kps2d_mask, -1))
    projector = triangulator.get_projector()
    projected_points = projector.project(keypoints3d)
    for cam_idx in range(n_view):
        canvas = cv2.imread(os.path.join(input_dir, f'{cam_idx:06}.png'))
        valid_idxs = np.where(kps2d_mask == 1)
        for point_idx in valid_idxs[1]:
            cv2.circle(
                img=canvas,
                center=projected_points[cam_idx, point_idx].astype(np.int32),
                radius=2,
                color=(0, 0, 255))
        cv2.imwrite(
            filename=os.path.join(output_dir,
                                  f'projected_aniposelib_{cam_idx}.jpg'),
            img=canvas)


def test_fourdag_triangulator():
    n_view = 6
    kps2d_list = []
    mask_list = []
    for view_idx in range(n_view):
        npz_path = os.path.join(input_dir, f'keypoints_2d_{view_idx:02d}.npz')
        npz_dict = dict(np.load(npz_path, allow_pickle=True))
        kps2d_list.append(npz_dict['keypoints'][0, 0, :, :])
        mask_list.append(npz_dict['mask'][0, 0, :])
    kps2d = np.asarray(kps2d_list)
    kps2d_mask = np.asarray(mask_list, dtype=kps2d.dtype)
    cam_param_list = []
    for view_idx in range(n_view):
        cam_param_path = os.path.join(input_dir, f'cam_{view_idx:03d}.json')
        cam_param = PinholeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    triangulator_config = dict(
        mmcv.Config.fromfile(
            'configs/modules/ops/triangulation/jacobi_triangulator.py'))
    triangulator_config['camera_parameters'] = cam_param_list
    triangulator = build_triangulator(triangulator_config)
    assert triangulator is not None
    # test kp2d np
    kps3d = triangulator.triangulate(kps2d)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test kp2d list
    kps3d = triangulator.triangulate(kps2d.tolist())
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test kp2d tuple
    kps3d = triangulator.triangulate(tuple(map(tuple, kps2d)))
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask np
    points_mask = np.ones_like(kps2d[..., 0:1])
    kps3d = triangulator.triangulate(points=kps2d, points_mask=points_mask)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask list
    kps3d = triangulator.triangulate(
        points=kps2d, points_mask=points_mask.tolist())
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    # test mask tuple
    kps3d = triangulator.triangulate(
        points=kps2d, points_mask=tuple(map(tuple, points_mask)))
    # test mask from confidence
    points_mask = kps2d_mask
    kps3d = triangulator.triangulate(points=kps2d, points_mask=points_mask)
    assert kps3d.shape[:2] == kps2d.shape[1:3]
    assert kps3d.shape[:2] == kps2d.shape[1:3]
