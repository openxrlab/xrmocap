import os
import pytest
import shutil
from xrprimer.data_structure import Keypoints

from xrmocap.core.evaluation.metric_manager import MetricManager
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

INPUT_DIR = 'tests/data/core/evaluation/test_metric_manager'
OUTPUT_DIR = 'tests/data/output/core/evaluation/test_metric_manager'
METRIC_LIST = [
    dict(
        type='PredictionMatcher',
        name='matching',
    ),
    dict(
        type='PCKMetric',
        name='pck_50',
        threshold=[50],
    ),
    dict(
        type='MPJPEMetric',
        name='mpjpe',
        align_kps_name='left_ankle',
    ),
    dict(
        type='PAMPJPEMetric',
        name='pa_mpjpe',
        align_kps_name='left_ankle',
    ),
]


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=False)


def test_construct():
    # test None pick_dict
    manager = MetricManager(
        metric_list=METRIC_LIST,
        pick_dict=None,
    )
    assert len(manager.pick_dict) == len(METRIC_LIST)
    # test one all
    manager = MetricManager(
        metric_list=METRIC_LIST,
        pick_dict=dict(pck_50='all'),
    )
    assert len(manager.pick_dict) == 1
    # test pick name and names
    manager = MetricManager(
        metric_list=METRIC_LIST,
        pick_dict=dict(
            pck_50='pck_value',
            mpjpe=[
                'mpjpe_value',
            ],
        ),
    )
    assert len(manager.pick_dict) == 2


def test_call():
    manager = MetricManager(
        metric_list=METRIC_LIST,
        pick_dict=dict(
            matching='all',
            pck_50='pck@50',
            mpjpe='mpjpe_mean',
            pa_mpjpe='pa_mpjpe_mean'),
    )
    gt_path = os.path.join(INPUT_DIR, 'gt_keypoints3d.npz')
    gt_keypoints3d = Keypoints.fromfile(gt_path)
    gt_keypoints3d = convert_keypoints(
        gt_keypoints3d, dst='coco', approximate=True)
    pred_path = os.path.join(INPUT_DIR, 'pred_keypoints3d.npz')
    pred_keypoints3d = Keypoints.fromfile(pred_path)
    pred_keypoints3d = convert_keypoints(
        pred_keypoints3d, dst='coco', approximate=True)
    result_dict, _ = manager(
        pred_keypoints3d=pred_keypoints3d,
        gt_keypoints3d=gt_keypoints3d,
    )
    assert len(result_dict) == len(METRIC_LIST)
