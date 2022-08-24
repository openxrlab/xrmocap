import cv2
import mmcv
import numpy as np
import os
import pytest
import shutil
from xrprimer.data_structure.camera import PinholeCameraParameter

from xrmocap.ops.projection.builder import build_projector

input_dir = 'tests/data/ops/test_projection'
output_dir = 'tests/data/output/ops/test_projection'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_aniposelib_projector():
    keypoints3d = np.load(
        os.path.join(input_dir, 'keypoints3d.npz'), allow_pickle=True)
    keypoints3d = dict(keypoints3d)
    n_view = 6
    cam_param_list = []
    for cam_idx in range(n_view):
        cam_param_path = os.path.join(input_dir, f'cam_{cam_idx:03d}.json')
        cam_param = PinholeCameraParameter()
        cam_param.load(cam_param_path)
        cam_param_list.append(cam_param)
    projector_config = dict(
        mmcv.Config.fromfile(
            'configs/modules/ops/projection/aniposelib_projector.py'))
    projector_config['camera_parameters'] = cam_param_list
    projector = build_projector(projector_config)
    point = np.array(keypoints3d['keypoints'][0, 0, 0, :3])
    # test project numpy
    projected_points = projector.project_single_point(point)
    assert projected_points.shape == (n_view, 2)
    # test project list
    projected_points = projector.project_single_point(point.tolist())
    assert projected_points.shape == (n_view, 2)
    # test project tuple
    projected_points = projector.project_single_point((0, 1, 2))
    assert projected_points.shape == (n_view, 2)
    points3d = keypoints3d['keypoints'][0, 0, :, :3]
    points3d_mask = np.expand_dims(keypoints3d['mask'][0, 0, :], axis=-1)
    n_point = points3d.shape[0]
    # test project numpy
    assert isinstance(points3d, np.ndarray)
    projected_points = projector.project(points3d)
    assert projected_points.shape == (n_view, n_point, 2)
    # test project list
    projected_points = projector.project(points3d.tolist())
    assert projected_points.shape == (n_view, n_point, 2)
    projected_points = projector.project(points3d, points_mask=points3d_mask)
    for cam_idx in range(n_view):
        canvas = cv2.imread(os.path.join(input_dir, f'{cam_idx:06}.png'))
        valid_idxs = np.where(points3d_mask == 1)
        for point_idx in valid_idxs[0]:
            cv2.circle(
                img=canvas,
                center=projected_points[cam_idx, point_idx].astype(np.int32),
                radius=2,
                color=(0, 0, 255))
        cv2.imwrite(
            filename=os.path.join(output_dir, f'projected_kps_{cam_idx}.jpg'),
            img=canvas)
