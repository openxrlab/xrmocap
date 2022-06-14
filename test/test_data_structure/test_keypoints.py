import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.keypoints import Keypoints

output_dir = 'test/data/output/test_data_structure/' +\
    'test_keypoints'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_new():
    # test __new__
    empty_keypoints = Keypoints()
    assert hasattr(empty_keypoints, 'logger')
    assert empty_keypoints.logger is not None
    # new with None
    keypoints = Keypoints()
    with pytest.raises(KeyError):
        keypoints.get_keypoints()
    # new with np values
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    assert keypoints.get_frame_number() == 2
    assert keypoints.get_person_number() == 3
    assert keypoints.get_keypoints_number() == 25
    # new with torch values
    kps_tensor = torch.zeros(size=(2, 3, 25, 3))
    mask_tensor = torch.zeros(size=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(
        kps=kps_tensor, mask=mask_tensor, convention=convention)
    assert isinstance(keypoints.get_keypoints(), torch.Tensor)
    assert keypoints.get_frame_number() == 2
    assert keypoints.get_person_number() == 3
    assert keypoints.get_keypoints_number() == 25
    # new with dict, while src_dict is torch
    # and dtype is auto
    src_dict = dict(keypoints)
    another_keypoints = Keypoints(src_dict=src_dict)
    assert isinstance(another_keypoints.get_keypoints(), torch.Tensor)
    assert isinstance(another_keypoints.get_mask(), torch.Tensor)
    assert isinstance(another_keypoints.get_convention(), str)


def test_set_convention():
    keypoints = Keypoints()
    keypoints.set_convention('coco')
    with pytest.raises(TypeError):
        keypoints.set_convention(133)


def test_set_keypoints():
    keypoints = Keypoints()
    keypoints.set_convention('smplx')
    # set np
    kps_np = np.zeros(shape=(2, 3, 144, 3))
    keypoints.set_keypoints(kps_np)
    assert keypoints.get_frame_number() == 2
    # set torch
    keypoints.set_keypoints(torch.from_numpy(kps_np))
    assert keypoints.get_frame_number() == 2
    # set one frame one person
    kps_np = np.zeros(shape=(144, 3))
    keypoints.set_keypoints(kps_np)
    assert keypoints.get_frame_number() == 1
    # set wrong type
    with pytest.raises(TypeError):
        keypoints.set_keypoints(kps_np.tolist())
    # set wrong dim
    kps_6d = np.zeros(shape=(2, 3, 144, 6))
    with pytest.raises(ValueError):
        keypoints.set_keypoints(kps_6d)
    # set wrong shape
    kps_np = np.zeros(shape=(3, 144, 6))
    with pytest.raises(ValueError):
        keypoints.set_keypoints(kps_np)


def test_convert_type():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    keypoints['n_kps'] = 25
    keypoints_tensor = keypoints.to_tensor(device='cpu')
    assert keypoints_tensor['convention'] == convention
    assert isinstance(keypoints_tensor['keypoints'], torch.Tensor)
    assert isinstance(keypoints_tensor['mask'], torch.Tensor)
    assert 'n_kps' not in keypoints_tensor
    keypoints_np = keypoints_tensor.to_numpy()
    assert keypoints_np['convention'] == convention
    assert isinstance(keypoints_np['keypoints'], np.ndarray)
    assert isinstance(keypoints_np['mask'], np.ndarray)
    assert 'n_kps' not in keypoints_np


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_convert_type_gpu():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    keypoints['n_kps'] = 25
    device_name = 'cuda:0'
    keypoints_tensor = keypoints.to_tensor(device=device_name)
    assert keypoints_tensor['convention'] == convention
    assert isinstance(keypoints_tensor['keypoints'], torch.Tensor)
    assert isinstance(keypoints_tensor['mask'], torch.Tensor)
    assert 'n_kps' not in keypoints_tensor


def test_file_io():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    keypoints['n_frame'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_keypoints2d.npz')
    keypoints.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    keypoints_new = Keypoints.fromfile(npz_path)
    assert keypoints_new['n_frame'] == 2
    assert keypoints_new.get_keypoints().shape == (2, 3, 25, 3)
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_keypoints2d.pkl')
    with pytest.raises(ValueError):
        keypoints.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        keypoints.dump(npz_path, overwrite=False)
    keypoints.dump(npz_path, overwrite=True)
    # test dump tensors to npz
    kps_tensor = torch.zeros(size=(2, 3, 25, 3))
    mask_tensor = torch.zeros(size=(2, 3, 25))
    convention = 'openpose_25'
    keypoints = Keypoints(
        kps=kps_tensor, mask=mask_tensor, convention=convention)
    assert isinstance(keypoints.get_keypoints(), torch.Tensor)
    keypoints.dump(npz_path, overwrite=True)
