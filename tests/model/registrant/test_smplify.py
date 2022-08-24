# yapf: disable
import mmcv
import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model.smpl_data import SMPLData
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
output_dir = 'tests/data/output/model/registrant/test_smplify'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_build():
    # normally build
    smplify_config = dict(
        mmcv.Config.fromfile('configs/modules/model/' +
                             'registrant/smplify_test.py'))
    smplify = build_registrant(smplify_config)
    assert smplify is not None
    # build with body_model
    body_model_smplify_config = smplify_config.copy()
    body_model_cfg = smplify_config['body_model']
    body_model = build_body_model(body_model_cfg)
    body_model_smplify_config['body_model'] = body_model
    smplify = build_registrant(body_model_smplify_config)
    assert smplify is not None
    # build with wrong type body_model
    body_model = 'smpl_body_model'
    body_model_smplify_config['body_model'] = body_model
    with pytest.raises(TypeError):
        smplify = build_registrant(body_model_smplify_config)
    # build with built handlers
    handler_smplify_config = smplify_config.copy()
    handlers = []
    for handler_cfg in smplify_config['handlers']:
        handlers.append(build_handler(handler_cfg))
    handler_smplify_config['handlers'] = handlers
    smplify = build_registrant(handler_smplify_config)
    assert smplify is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_smplify_keypoints3d():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    keypoints3d_path = os.path.join(input_dir, 'human_data_tri.npz')
    human_data = dict(np.load(keypoints3d_path, allow_pickle=True))
    keypoints3d, keypoints3d_mask = convert_kps_mm(
        keypoints=human_data['keypoints3d'][:2, :, :3],
        src='human_data',
        dst='smpl',
        mask=human_data['keypoints3d_mask'])
    keypoints3d = torch.from_numpy(keypoints3d).to(
        dtype=torch.float32, device=device)
    keypoints3d_conf = torch.from_numpy(np.expand_dims(
        keypoints3d_mask, 0)).to(
            dtype=torch.float32, device=device).repeat(keypoints3d.shape[0], 1)
    # build and run
    smplify_config = dict(
        mmcv.Config.fromfile('configs/modules/model/' +
                             'registrant/smplify_test.py'))
    smplify = build_registrant(smplify_config)
    kp3d_mse_input = Keypoint3dMSEInput(
        keypoints3d=keypoints3d,
        keypoints3d_conf=keypoints3d_conf,
        keypoints3d_convention='smpl',
        handler_key='keypoints3d_mse')
    kp3d_llen_input = Keypoint3dLimbLenInput(
        keypoints3d=keypoints3d,
        keypoints3d_conf=keypoints3d_conf,
        keypoints3d_convention='smpl',
        handler_key='keypoints3d_limb_len')
    smplify_output = smplify(input_list=[kp3d_mse_input, kp3d_llen_input])

    smpl_data = SMPLData()
    for k, v in smplify_output.items():
        if isinstance(v, torch.Tensor):
            np_v = v.detach().cpu().numpy()
            assert not np.any(np.isnan(np_v)), f'{k} fails.'
    smpl_data.from_param_dict(smplify_output)
    result_path = os.path.join(output_dir, 'smpl_result.npz')
    smpl_data.dump(result_path)
    # test not use_one_betas_per_video and return values
    m_betas_config = smplify_config.copy()
    m_betas_config['use_one_betas_per_video'] = False
    smplify = build_registrant(m_betas_config)
    smplify_output = smplify(
        input_list=[kp3d_mse_input, kp3d_llen_input],
        return_verts=True,
        return_joints=True,
        return_full_pose=True,
        return_losses=True)
    assert 'vertices' in smplify_output
    assert 'full_pose' in smplify_output
    assert 'joints' in smplify_output
    assert 'total_loss' in smplify_output
