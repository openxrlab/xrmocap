# yapf: disable
import mmcv
import os
import pytest
import shutil

from xrmocap.ops.bottom_up_association.builder import (
    build_bottom_up_associator,
)

# yapf: enable

output_dir = 'tests/data/output/ops/test_bottom_up_association'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


def test_build_bottom_up_associator():
    associator_cfg = dict(
        mmcv.Config.fromfile('configs/modules/ops/' +
                             'bottom_up_association/' +
                             'bottom_up_association.py'))
    associator = build_bottom_up_associator(associator_cfg)
    assert associator is not None
