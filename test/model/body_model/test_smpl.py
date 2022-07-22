import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model.smpl_data import SMPLData
from xrmocap.model.body_model.builder import build_body_model

body_model_load_dir = 'xrmocap_data/body_models/smpl'
extra_joints_regressor_path = 'xrmocap_data/body_models/J_regressor_extra.npy'
input_dir = 'test/data/model/body_model/test_smpl'
output_dir = 'test/data/output/model/body_model/test_smpl'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_smpl():
    random_body_pose = torch.rand((1, 69))
    # test SMPL without extra_joints_regressor
    smpl_45 = build_body_model(
        dict(
            type='SMPL',
            keypoint_convention='smpl_45',
            model_path=body_model_load_dir))
    smpl_45 = smpl_45(body_pose=random_body_pose)
    assert isinstance(smpl_45['joints'], torch.Tensor)
    assert smpl_45['joints'].shape[1] == 45
    smpl_data = SMPLData()
    smpl_data.from_param_dict(smpl_45)
    assert 'fullpose' in smpl_data
    assert isinstance(smpl_data['fullpose'], np.ndarray)
    npz_path = os.path.join(output_dir, 'dumped_smpl_data.npz')
    smpl_data.dump(npz_path)
    # test SMPL with extra_joints_regressor
    smpl_54 = build_body_model(
        dict(
            type='SMPL',
            keypoint_convention='smpl_54',
            model_path=body_model_load_dir,
            extra_joints_regressor=extra_joints_regressor_path))
    smpl_54_output = smpl_54(body_pose=random_body_pose)
    assert isinstance(smpl_54_output['joints'], torch.Tensor)
    assert smpl_54_output['joints'].shape[1] == 54
