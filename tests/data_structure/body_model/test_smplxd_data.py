import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model import SMPLXDData, auto_load_smpl_data

output_dir = 'tests/data/output/data_structure/' +\
    'body_model/test_smplxd_data'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_new():
    # empty new
    smplxd_data = SMPLXDData()
    assert hasattr(smplxd_data, 'logger')
    assert smplxd_data.logger is not None
    assert smplxd_data.get_fullpose().shape[1] == 55
    # new with specific value
    smplxd_data = SMPLXDData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        mask=np.ones(shape=(2, )),
        displacement=np.zeros(shape=(2, 10475)),
        logger='root')
    assert smplxd_data['betas'][0, 0] == 0
    assert smplxd_data.get_expression().shape == (2, 10)
    # new with smplxd_data_dict
    smplxd_data['betas'][0, 0] = 1
    smplxd_data_dict = dict(smplxd_data)
    smplxd_data = SMPLXDData.from_dict(smplxd_data_dict)
    assert smplxd_data['betas'][0, 0] == 1


def test_set_gender():
    smplxd_data = SMPLXDData()
    # set correct gender
    smplxd_data.set_gender('neutral')
    smplxd_data.set_gender('male')
    smplxd_data.set_gender('female')
    # set wrong gender
    with pytest.raises(ValueError):
        smplxd_data.set_gender('man')


def test_set_betas():
    smplxd_data = SMPLXDData()
    # set numpy
    smplxd_data.set_betas(np.zeros(shape=[2, 10]))
    # set torch
    smplxd_data.set_betas(torch.zeros(size=[2, 10]))
    # set one dim
    smplxd_data.set_betas(torch.zeros(size=[10]))
    # wrong type
    with pytest.raises(TypeError):
        smplxd_data.set_betas(np.zeros(shape=[2, 10]).tolist())


def test_set_transl():
    smplxd_data = SMPLXDData()
    # set numpy
    smplxd_data.set_transl(np.zeros(shape=[2, 3]))
    # set torch
    smplxd_data.set_transl(torch.zeros(size=[2, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smplxd_data.set_transl(np.zeros(shape=[2, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smplxd_data.set_transl(torch.zeros(size=[2, 4]))


def test_set_fullpose():
    smplxd_data = SMPLXDData()
    # set numpy
    smplxd_data.set_fullpose(np.zeros(shape=[2, 55, 3]))
    # set torch
    smplxd_data.set_fullpose(torch.zeros(size=[2, 55, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smplxd_data.set_fullpose(np.zeros(shape=[2, 55, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smplxd_data.set_fullpose(torch.zeros(size=[2, 55, 4]))


def test_setitem():
    smplxd_data = SMPLXDData()
    # set necessary parameter
    smplxd_data['betas'] = np.zeros(shape=[2, 10])
    smplxd_data['transl'] = np.zeros(shape=[2, 3])
    smplxd_data['fullpose'] = np.zeros(shape=[2, 55, 3])
    smplxd_data['gender'] = 'neutral'
    smplxd_data['displacement'] = np.zeros(shape=(2, 10475))
    # set arbitrary key
    smplxd_data['frame_number'] = 1000


def test_dict_io():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    smplxd_data = SMPLXDData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)),
        expression=np.zeros(shape=(10)))
    smplxd_data['frame_num'] = 2
    assert smplxd_data['betas'].shape == (1, 10)
    assert smplxd_data['expression'].shape == (1, 10)
    # test repeat betas
    param_dict = smplxd_data.to_param_dict(repeat_betas=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (1, 10)
    assert 'frame_num' not in param_dict
    param_dict = smplxd_data.to_param_dict(repeat_betas=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (2, 10)
    assert 'frame_num' not in param_dict
    # test repeat expression
    param_dict = smplxd_data.to_param_dict(repeat_expression=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['expression'].shape == (1, 10)
    assert 'frame_num' not in param_dict
    param_dict = smplxd_data.to_param_dict(repeat_expression=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['expression'].shape == (2, 10)
    assert 'frame_num' not in param_dict
    # test to tensor
    tensor_dict = smplxd_data.to_tensor_dict(
        repeat_betas=True, repeat_expression=True, device=device_name)
    assert isinstance(tensor_dict['global_orient'], torch.Tensor)
    assert tensor_dict['global_orient'].shape == (2, 3)
    # test from numpy dict
    smplxd_data_new = SMPLXDData()
    smplxd_data_new.from_param_dict(param_dict)
    assert smplxd_data_new['fullpose'].shape == (2, 55, 3)
    # test from torch dict
    smplxd_data_new = SMPLXDData()
    smplxd_data_new.from_param_dict(tensor_dict)
    assert smplxd_data_new['fullpose'].shape == (2, 55, 3)
    # test
    with pytest.raises(KeyError):
        param_dict.pop('global_orient')
        smplxd_data_new = SMPLXDData()
        smplxd_data_new.from_param_dict(param_dict)


def test_file_io():
    smplxd_data = SMPLXDData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)),
        expression=np.zeros(shape=(10)))
    smplxd_data['frame_num'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_smplxd_data.npz')
    smplxd_data.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    smplxd_data_new = SMPLXDData.fromfile(npz_path)
    assert smplxd_data_new['frame_num'] == 2
    assert smplxd_data_new.get_fullpose().shape == (2, 55, 3)
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_smplxd_data.pkl')
    with pytest.raises(ValueError):
        smplxd_data.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        smplxd_data.dump(npz_path, overwrite=False)
    smplxd_data.dump(npz_path, overwrite=True)
    # test auto load
    instance, class_name = auto_load_smpl_data(npz_path)
    assert isinstance(instance, SMPLXDData)
    assert class_name == 'SMPLXDData'
