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
    empty_kps = Keypoints()
    assert hasattr(empty_kps, 'logger')
    assert empty_kps.logger is not None
    # new with None
    kps = Keypoints()
    with pytest.raises(KeyError):
        kps.get_keypoints()
    # new with np values
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    assert kps.get_frame_number() == 2
    assert kps.get_person_number() == 3
    assert kps.get_keypoints_number() == 25
    # new with torch values
    kps_tensor = torch.zeros(size=(2, 3, 25, 3))
    mask_tensor = torch.zeros(size=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_tensor, mask=mask_tensor, convention=convention)
    assert isinstance(kps.get_keypoints(), torch.Tensor)
    assert kps.get_frame_number() == 2
    assert kps.get_person_number() == 3
    assert kps.get_keypoints_number() == 25
    # new with dict, while src_dict is torch
    # and dtype is auto
    src_dict = dict(kps)
    another_kps = Keypoints(src_dict=src_dict)
    assert isinstance(another_kps.get_keypoints(), torch.Tensor)
    assert isinstance(another_kps.get_mask(), torch.Tensor)
    assert isinstance(another_kps.get_convention(), str)


def test_set_convention():
    kps = Keypoints()
    kps.set_convention('coco')
    with pytest.raises(TypeError):
        kps.set_convention(133)


def test_set_keypoints():
    kps = Keypoints()
    kps.set_convention('smplx')
    # set np
    kps_np = np.zeros(shape=(2, 3, 144, 3))
    kps.set_keypoints(kps_np)
    assert kps.get_frame_number() == 2
    # set torch
    kps.set_keypoints(torch.from_numpy(kps_np))
    assert kps.get_frame_number() == 2
    # set one frame one person
    kps_np = np.zeros(shape=(144, 3))
    kps.set_keypoints(kps_np)
    assert kps.get_frame_number() == 1
    # set wrong type
    with pytest.raises(TypeError):
        kps.set_keypoints(kps_np.tolist())
    # set wrong dim
    kps_6d = np.zeros(shape=(2, 3, 144, 6))
    with pytest.raises(ValueError):
        kps.set_keypoints(kps_6d)
    # set wrong shape
    kps_np = np.zeros(shape=(3, 144, 6))
    with pytest.raises(ValueError):
        kps.set_keypoints(kps_np)


def test_convert_type():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    kps['kp_num'] = 25
    kps_tensor = kps.to_tensor(device='cpu')
    assert kps_tensor['convention'] == convention
    assert isinstance(kps_tensor['keypoints'], torch.Tensor)
    assert isinstance(kps_tensor['mask'], torch.Tensor)
    assert 'kp_num' not in kps_tensor
    kps_np = kps_tensor.to_numpy()
    assert kps_np['convention'] == convention
    assert isinstance(kps_np['keypoints'], np.ndarray)
    assert isinstance(kps_np['mask'], np.ndarray)
    assert 'kp_num' not in kps_np


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_convert_type_gpu():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    kps['kp_num'] = 25
    device_name = 'cuda:0'
    kps_tensor = kps.to_tensor(device=device_name)
    assert kps_tensor['convention'] == convention
    assert isinstance(kps_tensor['keypoints'], torch.Tensor)
    assert isinstance(kps_tensor['mask'], torch.Tensor)
    assert 'kp_num' not in kps_tensor


def test_file_io():
    kps_np = np.zeros(shape=(2, 3, 25, 3))
    mask_np = np.zeros(shape=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_np, mask=mask_np, convention=convention)
    kps['frame_num'] = 2
    # test correctly dump
    npz_path = os.path.join(output_dir, 'dumped_kps2d.npz')
    kps.dump(npz_path)
    assert os.path.exists(npz_path)
    # test correctly load
    kps_new = Keypoints.fromfile(npz_path)
    assert kps_new['frame_num'] == 2
    assert kps_new.get_keypoints().shape == (2, 3, 25, 3)
    # test wrong filename
    pkl_path = os.path.join(output_dir, 'dumped_kps2d.pkl')
    with pytest.raises(ValueError):
        kps.dump(pkl_path)
    # test overwrite
    with pytest.raises(FileExistsError):
        kps.dump(npz_path, overwrite=False)
    kps.dump(npz_path, overwrite=True)
    # test dump tensors to npz
    kps_tensor = torch.zeros(size=(2, 3, 25, 3))
    mask_tensor = torch.zeros(size=(2, 3, 25))
    convention = 'openpose_25'
    kps = Keypoints(kps=kps_tensor, mask=mask_tensor, convention=convention)
    assert isinstance(kps.get_keypoints(), torch.Tensor)
    kps.dump(npz_path, overwrite=True)
