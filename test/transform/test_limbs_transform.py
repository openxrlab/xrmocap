import cv2
import numpy as np
import os
import pytest
import shutil

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.limbs import get_limbs_from_keypoints

input_dir = 'test/data/transform/test_limbs'
output_dir = 'test/data/output/transform/test_limbs'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_get_limbs_from_keypoints():
    # test get from numpy
    keypoints3d = Keypoints.fromfile(
        'test/data/ops/test_projection/keypoints3d.npz')
    limbs = get_limbs_from_keypoints(keypoints=keypoints3d)
    assert len(limbs) > 0
    assert limbs.get_points() is None
    # test get from torch
    keypoints3d_torch = keypoints3d.to_tensor()
    limbs = get_limbs_from_keypoints(keypoints=keypoints3d_torch)
    assert len(limbs) > 0
    assert len(limbs.get_parts()) > 0
    assert limbs.get_points() is None
    # test get with points
    limbs = get_limbs_from_keypoints(
        keypoints=keypoints3d, frame_idx=0, person_idx=0)
    assert limbs.get_points() is not None
    conn = limbs.get_connections()
    canvas = np.ones(shape=(1080, 1920, 3), dtype=np.uint8)
    points = limbs.get_points()
    for start_pt_idx, end_pt_idx in conn:
        cv2.line(
            img=canvas,
            pt1=points[start_pt_idx, :2].astype(np.int32),
            pt2=points[end_pt_idx, :2].astype(np.int32),
            color=(255, 0, 0),
            thickness=2)
    cv2.imwrite(
        filename=os.path.join(output_dir, 'limbs_from_keypoints.jpg'),
        img=canvas)
    # test get connection names
    limbs = get_limbs_from_keypoints(
        keypoints=keypoints3d, frame_idx=0, person_idx=0, fill_limb_names=True)
    conn_dict = limbs.get_connections_by_names()
    assert len(conn_dict) > 0
