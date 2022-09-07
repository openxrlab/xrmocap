import mmcv
import os
import pytest
import shutil

from xrmocap.ops.fourd_association.builder import build_fourd_associator

output_dir = 'tests/data/output/ops/test_fourd_association'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_run_mvpose_associator():
    associator_cfg = dict(
        mmcv.Config.fromfile('configs/modules/ops/' + 'fourd_association/' +
                             'fourd_association.py'))
    associator = build_fourd_associator(associator_cfg)
    assert associator is not None


