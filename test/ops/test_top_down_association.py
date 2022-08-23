import mmcv
import numpy as np
import os
import os.path as osp
import pytest
import shutil

from xrmocap.evaluation.builder import build_evaluation

input_dir = 'test/data/keypoints3d_estimation/'
output_dir = 'test/data/output/keypoints3d_estimation/shelf'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_top_down_association():
    evaluation_config = dict(
        mmcv.Config.fromfile(
            './config/ops/top_down_association/top_down_associator.py'))
    evaluation_config['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    evaluation = build_evaluation(evaluation_config)
    evaluation.run(overwrite=True)
    matched_kps2d_idx = np.load(
        osp.join(output_dir, 'scene0_matched_kps2d_idx.npy'))
    assert matched_kps2d_idx.shape == (5, 2, 5)
    assert matched_kps2d_idx[0].all() == np.array([[0., 1., 0., 0., 1.],
                                                   [1., 0., 1., 1.,
                                                    0.]]).all()
