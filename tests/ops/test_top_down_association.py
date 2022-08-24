import mmcv
import os
import pytest
import shutil
import torch

from xrmocap.ops.top_down_association.builder import build_top_down_associator

output_dir = 'tests/data/output/ops/test_top_down_association'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_build_mvpose_associator():
    associator_cfg = dict(
        mmcv.Config.fromfile('configs/modules/ops/' + 'top_down_association/' +
                             'mvpose_tracking_associator.py'))
    os.makedirs(output_dir, exist_ok=True)
    associator = build_top_down_associator(associator_cfg)
    assert associator is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_run_mvpose_associator():
    associator_cfg = dict(
        mmcv.Config.fromfile('configs/modules/ops/' + 'top_down_association/' +
                             'mvpose_associator.py'))
    os.makedirs(output_dir, exist_ok=True)
    associator = build_top_down_associator(associator_cfg)
    assert associator is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_run_mvpose_tracking_associator():
    associator_cfg = dict(
        mmcv.Config.fromfile('configs/modules/ops/' + 'top_down_association/' +
                             'mvpose_tracking_associator.py'))
    os.makedirs(output_dir, exist_ok=True)
    associator = build_top_down_associator(associator_cfg)
    assert associator is not None
