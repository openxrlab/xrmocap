import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model.smpl_data import SMPLData

output_dir = 'test/data/output/test_data_structure/' +\
    'test_body_model/test_smpl_data'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_new():
    # empty new
    smpl_data = SMPLData()
    assert hasattr(smpl_data, 'logger')
    assert smpl_data.logger is not None
    # new from dict
    smpl_data = SMPLData({'frame_num': 20})
    assert smpl_data['frame_num'] == 20
    # new with specific value
    smpl_data = SMPLData(
        gender='neutral',
        full_pose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        logger='root')
    assert smpl_data['betas'][0, 0] == 0
    # new with source dict
    src_dict = {'frame_num': 2, 'betas': np.ones(shape=(2, 10))}
    smpl_data = SMPLData(
        src_dict=src_dict,
        gender='neutral',
        full_pose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        logger='root')
    assert smpl_data['betas'][0, 0] == 0


def test_set_gender():
    smpl_data = SMPLData()
    # set correct gender
    smpl_data.set_gender('neutral')
    smpl_data.set_gender('male')
    smpl_data.set_gender('female')
    # set wrong gender
    with pytest.raises(ValueError):
        smpl_data.set_gender('man')


def test_set_betas():
    smpl_data = SMPLData()
    # set numpy
    smpl_data.set_betas(np.zeros(shape=[2, 10]))
    # set torch
    smpl_data.set_betas(torch.zeros(size=[2, 10]))
    # set one dim
    smpl_data.set_betas(torch.zeros(size=[10]))
    # wrong type
    with pytest.raises(TypeError):
        smpl_data.set_betas(np.zeros(shape=[2, 10]).tolist())


def test_set_transl():
    smpl_data = SMPLData()
    # set numpy
    smpl_data.set_transl(np.zeros(shape=[2, 3]))
    # set torch
    smpl_data.set_transl(torch.zeros(size=[2, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smpl_data.set_transl(np.zeros(shape=[2, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smpl_data.set_transl(torch.zeros(size=[2, 4]))


def test_set_fullpose():
    smpl_data = SMPLData()
    # set numpy
    smpl_data.set_fullpose(np.zeros(shape=[2, 24, 3]))
    # set torch
    smpl_data.set_fullpose(torch.zeros(size=[2, 24, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smpl_data.set_fullpose(np.zeros(shape=[2, 24, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smpl_data.set_fullpose(torch.zeros(size=[2, 24, 4]))


def test_setitem():
    smpl_data = SMPLData()
    # set necessary parameter
    smpl_data['betas'] = np.zeros(shape=[2, 10])
    smpl_data['transl'] = np.zeros(shape=[2, 3])
    smpl_data['fullpose'] = np.zeros(shape=[2, 24, 3])
    smpl_data['gender'] = 'neutral'
    # set arbitrary key
    smpl_data['frame_number'] = 1000


def test_dict_io():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    smpl_data = SMPLData(
        gender='neutral',
        full_pose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)))
    smpl_data['frame_num'] = 2
    assert smpl_data['betas'].shape == (1, 10)
    # test default to dict
    param_dict = smpl_data.to_param_dict(repeat_betas=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (1, 10)
    assert 'frame_num' not in param_dict
    # test repeat betas
    param_dict = smpl_data.to_param_dict(repeat_betas=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (2, 10)
    assert 'frame_num' not in param_dict
    # test to tensor
    tensor_dict = smpl_data.to_tensor_dict(
        repeat_betas=True, device=device_name)
    assert isinstance(tensor_dict['global_orient'], torch.Tensor)
    assert tensor_dict['global_orient'].shape == (2, 3)
    # test from numpy dict
    smpl_data_new = SMPLData()
    smpl_data_new.from_param_dict(param_dict)
    assert smpl_data_new['full_pose'].shape == (2, 24, 3)
    # test from torch dict
    smpl_data_new = SMPLData()
    smpl_data_new.from_param_dict(tensor_dict)
    assert smpl_data_new['full_pose'].shape == (2, 24, 3)
    # test
    with pytest.raises(KeyError):
        param_dict.pop('global_orient')
        smpl_data_new = SMPLData()
        smpl_data_new.from_param_dict(param_dict)


def test_file_io():
    smpl_data = SMPLData(
        gender='neutral',
        full_pose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)))
    smpl_data['frame_num'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_smpl_data.npz')
    smpl_data.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    smpl_data_new = SMPLData.fromfile(npz_path)
    assert smpl_data_new['frame_num'] == 2
    assert smpl_data_new['full_pose'].shape == (2, 24, 3)
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_smpl_data.pkl')
    with pytest.raises(ValueError):
        smpl_data.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        smpl_data.dump(npz_path, overwrite=False)
    smpl_data.dump(npz_path, overwrite=True)
