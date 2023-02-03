import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model import SMPLData

output_dir = 'tests/data/output/data_structure/' +\
    'body_model/test_smpl_data'


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
    # new with specific value
    smpl_data = SMPLData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        mask=np.ones(shape=(2, )),
        logger='root')
    assert smpl_data['betas'][0, 0] == 0
    # new with smpl_dict
    smpl_data['betas'][0, 0] = 1
    smpl_data_dict = dict(smpl_data)
    smpl_data = SMPLData.from_dict(smpl_data_dict)
    assert smpl_data['betas'][0, 0] == 1


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
    smpl_data['n_frame'] = 1000


def test_dict_io():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    smpl_data = SMPLData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)))
    smpl_data['n_frame'] = 2
    assert smpl_data['betas'].shape == (1, 10)
    # test default to dict
    param_dict = smpl_data.to_param_dict(repeat_betas=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (1, 10)
    assert 'n_frame' not in param_dict
    # test repeat betas
    param_dict = smpl_data.to_param_dict(repeat_betas=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (2, 10)
    assert 'n_frame' not in param_dict
    # test to tensor
    tensor_dict = smpl_data.to_tensor_dict(
        repeat_betas=True, device=device_name)
    assert isinstance(tensor_dict['global_orient'], torch.Tensor)
    assert tensor_dict['global_orient'].shape == (2, 3)
    # test from numpy dict
    smpl_data_new = SMPLData()
    smpl_data_new.from_param_dict(param_dict)
    assert smpl_data_new['fullpose'].shape == (2, 24, 3)
    # test from torch dict
    smpl_data_new = SMPLData()
    smpl_data_new.from_param_dict(tensor_dict)
    assert smpl_data_new['fullpose'].shape == (2, 24, 3)
    # test
    with pytest.raises(KeyError):
        param_dict.pop('global_orient')
        smpl_data_new = SMPLData()
        smpl_data_new.from_param_dict(param_dict)


def test_file_io():
    smpl_data = SMPLData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 24, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)))
    smpl_data['n_frame'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_smpl_data.npz')
    smpl_data.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    smpl_data_new = SMPLData.fromfile(npz_path)
    assert smpl_data_new.get_fullpose().shape == (2, 24, 3)
    assert smpl_data_new['n_frame'] == 2
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_smpl_data.pkl')
    with pytest.raises(ValueError):
        smpl_data.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        smpl_data.dump(npz_path, overwrite=False)
    smpl_data.dump(npz_path, overwrite=True)
