import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.body_model import SMPLXData

output_dir = 'test/data/output/data_structure/' +\
    'body_model/test_smplx_data'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_new():
    # empty new
    smplx_data = SMPLXData()
    assert hasattr(smplx_data, 'logger')
    assert smplx_data.logger is not None
    assert smplx_data.get_fullpose().shape[1] == 55
    # new from dict
    smplx_data = SMPLXData({'frame_num': 20})
    assert smplx_data['frame_num'] == 20
    # new with specific value
    smplx_data = SMPLXData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        logger='root')
    assert smplx_data['betas'][0, 0] == 0
    assert smplx_data.get_expression().shape == (2, 10)
    # new with source dict
    src_dict = {'frame_num': 2, 'betas': np.ones(shape=(2, 10))}
    smplx_data = SMPLXData(
        src_dict=src_dict,
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(2, 10)),
        logger='root')
    assert smplx_data['betas'][0, 0] == 0


def test_set_gender():
    smplx_data = SMPLXData()
    # set correct gender
    smplx_data.set_gender('neutral')
    smplx_data.set_gender('male')
    smplx_data.set_gender('female')
    # set wrong gender
    with pytest.raises(ValueError):
        smplx_data.set_gender('man')


def test_set_betas():
    smplx_data = SMPLXData()
    # set numpy
    smplx_data.set_betas(np.zeros(shape=[2, 10]))
    # set torch
    smplx_data.set_betas(torch.zeros(size=[2, 10]))
    # set one dim
    smplx_data.set_betas(torch.zeros(size=[10]))
    # wrong type
    with pytest.raises(TypeError):
        smplx_data.set_betas(np.zeros(shape=[2, 10]).tolist())


def test_set_transl():
    smplx_data = SMPLXData()
    # set numpy
    smplx_data.set_transl(np.zeros(shape=[2, 3]))
    # set torch
    smplx_data.set_transl(torch.zeros(size=[2, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smplx_data.set_transl(np.zeros(shape=[2, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smplx_data.set_transl(torch.zeros(size=[2, 4]))


def test_set_fullpose():
    smplx_data = SMPLXData()
    # set numpy
    smplx_data.set_fullpose(np.zeros(shape=[2, 55, 3]))
    # set torch
    smplx_data.set_fullpose(torch.zeros(size=[2, 55, 3]))
    # wrong type
    with pytest.raises(TypeError):
        smplx_data.set_fullpose(np.zeros(shape=[2, 55, 3]).tolist())
    # wrong shape
    with pytest.raises(ValueError):
        smplx_data.set_fullpose(torch.zeros(size=[2, 55, 4]))


def test_setitem():
    smplx_data = SMPLXData()
    # set necessary parameter
    smplx_data['betas'] = np.zeros(shape=[2, 10])
    smplx_data['transl'] = np.zeros(shape=[2, 3])
    smplx_data['fullpose'] = np.zeros(shape=[2, 55, 3])
    smplx_data['gender'] = 'neutral'
    # set arbitrary key
    smplx_data['frame_number'] = 1000


def test_dict_io():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    smplx_data = SMPLXData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)),
        expression=np.zeros(shape=(10)))
    smplx_data['frame_num'] = 2
    assert smplx_data['betas'].shape == (1, 10)
    assert smplx_data['expression'].shape == (1, 10)
    # test repeat betas
    param_dict = smplx_data.to_param_dict(repeat_betas=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (1, 10)
    assert 'frame_num' not in param_dict
    param_dict = smplx_data.to_param_dict(repeat_betas=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['betas'].shape == (2, 10)
    assert 'frame_num' not in param_dict
    # test repeat expression
    param_dict = smplx_data.to_param_dict(repeat_expression=False)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['expression'].shape == (1, 10)
    assert 'frame_num' not in param_dict
    param_dict = smplx_data.to_param_dict(repeat_expression=True)
    assert param_dict['global_orient'].shape == (2, 3)
    assert param_dict['expression'].shape == (2, 10)
    assert 'frame_num' not in param_dict
    # test to tensor
    tensor_dict = smplx_data.to_tensor_dict(
        repeat_betas=True, repeat_expression=True, device=device_name)
    assert isinstance(tensor_dict['global_orient'], torch.Tensor)
    assert tensor_dict['global_orient'].shape == (2, 3)
    # test from numpy dict
    smplx_data_new = SMPLXData()
    smplx_data_new.from_param_dict(param_dict)
    assert smplx_data_new['fullpose'].shape == (2, 55, 3)
    # test from torch dict
    smplx_data_new = SMPLXData()
    smplx_data_new.from_param_dict(tensor_dict)
    assert smplx_data_new['fullpose'].shape == (2, 55, 3)
    # test
    with pytest.raises(KeyError):
        param_dict.pop('global_orient')
        smplx_data_new = SMPLXData()
        smplx_data_new.from_param_dict(param_dict)


def test_file_io():
    smplx_data = SMPLXData(
        gender='neutral',
        fullpose=np.zeros(shape=(2, 55, 3)),
        transl=np.zeros(shape=(2, 3)),
        betas=np.zeros(shape=(10)),
        expression=np.zeros(shape=(10)))
    smplx_data['frame_num'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_smplx_data.npz')
    smplx_data.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    smplx_data_new = SMPLXData.fromfile(npz_path)
    assert smplx_data_new['frame_num'] == 2
    assert smplx_data_new.get_fullpose().shape == (2, 55, 3)
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_smplx_data.pkl')
    with pytest.raises(ValueError):
        smplx_data.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        smplx_data.dump(npz_path, overwrite=False)
    smplx_data.dump(npz_path, overwrite=True)
