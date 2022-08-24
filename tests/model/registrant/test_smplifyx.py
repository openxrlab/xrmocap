# yapf: disable
import mmcv
import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model.smplx_data import SMPLXData
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.model.registrant.builder import build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.model.registrant.handler.keypoint3d_limb_length_handler import (
    Keypoint3dLimbLenInput,
)
from xrmocap.model.registrant.handler.keypoint3d_mse_handler import (
    Keypoint3dMSEInput,
)
from xrmocap.transform.convention.keypoints_convention import convert_kps_mm

# yapf: enable
input_dir = 'tests/data/model/registrant'
output_dir = 'tests/data/output/model/registrant/test_smplifyx'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_build():
    # normally build
    smplifyx_config = dict(
        mmcv.Config.fromfile(
            'configs/modules/model/registrant/smplifyx_test.py'))
    smplifyx = build_registrant(smplifyx_config)
    assert smplifyx is not None
    # build with body_model
    body_model_smplifyx_config = smplifyx_config.copy()
    body_model_cfg = smplifyx_config['body_model']
    body_model = build_body_model(body_model_cfg)
    body_model_smplifyx_config['body_model'] = body_model
    smplifyx = build_registrant(body_model_smplifyx_config)
    assert smplifyx is not None
    # build with wrong type body_model
    body_model = 'smpl_body_model'
    body_model_smplifyx_config['body_model'] = body_model
    with pytest.raises(TypeError):
        smplifyx = build_registrant(body_model_smplifyx_config)
    # build with built handlers
    handler_smplifyx_config = smplifyx_config.copy()
    handlers = []
    for handler_cfg in smplifyx_config['handlers']:
        handlers.append(build_handler(handler_cfg))
    handler_smplifyx_config['handlers'] = handlers
    smplifyx = build_registrant(handler_smplifyx_config)
    assert smplifyx is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_smplifyx_kps3d():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    kps3d_path = os.path.join(input_dir, 'human_data_tri.npz')
    human_data = dict(np.load(kps3d_path, allow_pickle=True))
    kps3d, kps3d_mask = convert_kps_mm(
        keypoints=human_data['keypoints3d'][:2, :, :3],
        src='human_data',
        dst='smplx',
        mask=human_data['keypoints3d_mask'])
    kps3d = torch.from_numpy(kps3d).to(dtype=torch.float32, device=device)
    kps3d_conf = torch.from_numpy(np.expand_dims(kps3d_mask, 0)).to(
        dtype=torch.float32, device=device).repeat(kps3d.shape[0], 1)
    # build and run
    smplifyx_config = dict(
        mmcv.Config.fromfile('configs/modules/model/' +
                             'registrant/smplifyx_test.py'))
    smplifyx = build_registrant(smplifyx_config)
    kp3d_mse_input = Keypoint3dMSEInput(
        keypoints3d=kps3d,
        keypoints3d_conf=kps3d_conf,
        keypoints3d_convention='smplx',
        handler_key='keypoints3d_mse')
    kp3d_llen_input = Keypoint3dLimbLenInput(
        keypoints3d=kps3d,
        keypoints3d_conf=kps3d_conf,
        keypoints3d_convention='smplx',
        handler_key='keypoints3d_limb_len')
    smplifyx_output = smplifyx(input_list=[kp3d_mse_input, kp3d_llen_input])

    smplx_data = SMPLXData()
    for k, v in smplifyx_output.items():
        if isinstance(v, torch.Tensor):
            np_v = v.detach().cpu().numpy()
            assert not np.any(np.isnan(np_v)), f'{k} fails.'
    smplx_data.from_param_dict(smplifyx_output)
    result_path = os.path.join(output_dir, 'smplx_result.npz')
    smplx_data.dump(result_path)
    # test not use_one_betas_per_video and return values
    m_betas_config = smplifyx_config.copy()
    m_betas_config['use_one_betas_per_video'] = False
    smplifyx = build_registrant(m_betas_config)
    smplifyx_output = smplifyx(
        input_list=[kp3d_mse_input, kp3d_llen_input],
        return_verts=True,
        return_joints=True,
        return_full_pose=True,
        return_losses=True)
    assert 'vertices' in smplifyx_output
    assert 'full_pose' in smplifyx_output
    assert 'joints' in smplifyx_output
    assert 'total_loss' in smplifyx_output
