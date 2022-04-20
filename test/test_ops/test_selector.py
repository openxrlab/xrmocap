import mmcv
import numpy as np

from xrmocap.ops.triangulation.builder import build_point_selector


def test_manual_threshold_selector():
    selector_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/manual_threshold_selector.py'))
    selector_config['threshold'] = 0.5
    selector = build_point_selector(selector_config)
    points2d = np.zeros((2, 17, 3))
    points2d_mask = selector.get_selection_mask(points2d)
    assert points2d_mask.sum() == 0.0
    points2d = np.zeros((2, 3, 17, 3))
    points2d += 0.6
    points2d_mask = selector.get_selection_mask(points2d)
    assert points2d_mask.sum() == 2 * 3 * 17
    points2d = np.zeros((48, 1, 3))
    points2d[:24, :] += 0.6
    points2d_mask = selector.get_selection_mask(points2d)
    assert points2d_mask.sum() == 24
    points2d_mask = selector.get_selection_mask(points2d.tolist())
    assert points2d_mask.sum() == 24
    points2d = np.zeros((48, 2, 3))
    points2d[:3, 0, :] += 0.6
    points2d[:2, 1, :] += 0.6
    points2d_mask = selector.get_selection_mask(points2d)
    assert points2d_mask[2, 0, 0] == 1
    assert points2d_mask[2, 1, 0] == 0
