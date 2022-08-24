import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model import SMPLXData
from xrmocap.model.body_model.builder import build_body_model

body_model_load_dir = 'xrmocap_data/body_models/smplx'
input_dir = 'tests/data/model/body_model/test_smplx'
output_dir = 'tests/data/output/model/body_model/test_smplx'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_smplx():
    random_body_pose = torch.rand((1, 63))
    left_hand_pose = torch.rand((1, 45))
    right_hand_pose = torch.rand((1, 45))
    jaw_pose = torch.rand((1, 3))
    leye_pose = torch.rand((1, 3))
    reye_pose = torch.rand((1, 3))
    # test SMPLX without extra_joints_regressor
    smplx = build_body_model(
        dict(
            type='SMPLX',
            gender='neutral',
            num_betas=10,
            use_face_contour=True,
            keypoint_convention='smplx',
            model_path=body_model_load_dir,
            batch_size=1,
            use_pca=False,
            logger=None))
    smplx_output = smplx(
        body_pose=random_body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose)
    assert isinstance(smplx_output['joints'], torch.Tensor)
    assert smplx_output['joints'].shape[1] == 144
    smplx_data = SMPLXData()
    smplx_data.from_param_dict(smplx_output)
    assert 'fullpose' in smplx_data
    assert isinstance(smplx_data['fullpose'], np.ndarray)
    npz_path = os.path.join(output_dir, 'dumped_smplx_data.npz')
    smplx_data.dump(npz_path)
