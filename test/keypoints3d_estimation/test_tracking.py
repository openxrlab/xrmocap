import mmcv
import os.path
import pytest
import shutil
import torch

from xrmocap.keypoints3d_estimation.estimation import Estimation

input_dir = 'test/data/keypoints3d_estimation/'
output_dir = 'test/data/output/keypoints3d_estimation/shelf'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_tracking():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    estimate_kps3d_config = mmcv.Config.fromfile(
        './config/kps3d_estimation/shelf_config/estimate_kps3d.py')
    estimate_kps3d_config.device = device_name
    estimate_kps3d_config.data['start_frame'] = 300
    estimate_kps3d_config.data['end_frame'] = 305
    estimate_kps3d_config.data['input_root'] = input_dir
    estimate_kps3d_config.output_dir = output_dir
    estimate_kps3d_config.use_kalman_tracking = True
    estimate_kps3d_config.camera_parameter_path = os.path.join(
        input_dir, 'shelf/omni.json')
    estimate_kps3d_config.homo_folder = os.path.join(input_dir,
                                                     'shelf/extrinsics')
    if device_name == 'cpu':
        return 0
    estimation = Estimation(estimate_kps3d_config)
    estimation.enable_camera()
    estimation.load_keypoints2d_data()
    keypoints3d = estimation.kalman_tracking_keypoints3d()
    kps = keypoints3d.get_keypoints()[..., :3]
    assert kps.shape == (5, 2, 17, 3)
