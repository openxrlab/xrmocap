import mmcv
import os
import os.path as osp
import pytest
import shutil

from xrmocap.core.evaluation.builder import build_evaluation
from xrmocap.data_structure.keypoints import Keypoints

output_dir = 'tests/data/output/core/test_fourdag_evaluation'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_fourdag_evaluation():
    evaluation_config = dict(
        mmcv.Config.fromfile('configs/modules/core/evaluation/' +
                             'bottom_up_eval_shelf_unittest.py'))
    evaluation_config['output_dir'] = output_dir
    evaluation_config['dataset_visualization']['output_dir'] = output_dir
    evaluation_config['dataset_visualization']['pred_kps3d_paths'] = osp.join(
        output_dir, 'scene0_pred_keypoints3d.npz')
    os.makedirs(output_dir, exist_ok=True)
    evaluation = build_evaluation(evaluation_config)
    evaluation.run(overwrite=True)
    pred_keypoints3d = Keypoints.fromfile(
        osp.join(output_dir, 'scene0_pred_keypoints3d.npz'))
    pred_kps3d = pred_keypoints3d.get_keypoints()
    assert pred_kps3d.shape == (5, 2, 19, 4)
    assert pred_keypoints3d.get_mask().shape == (5, 2, 19)
