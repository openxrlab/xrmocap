import mmcv
import numpy as np

from xrmocap.ops.triangulation.point_selection.builder import \
    build_point_selector  # prevent linting conflicts


def test_manual_threshold_selector():
    selector_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/manual_threshold_selector.py'))
    selector_config['threshold'] = 0.5
    selector = build_point_selector(selector_config)
    # test input not changed
    points2d = np.zeros((2, 17, 3))
    points2d[0, 0, 2] = 0.4
    points2d[0, 1, 2] = 0.7
    points2d[0, 2, 2] = 1.0
    init_mask = np.ones((2, 17, 1))
    init_mask[:, 3, :] = np.nan
    points2d_backup = points2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = selector.get_selection_mask(points2d, init_mask)
    assert np.all(points2d_backup == points2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    # test selection results
    assert points2d_mask[0, 0, 0] == 0.0
    assert points2d_mask[0, 1, 0] == 1.0
    assert points2d_mask[0, 2, 0] == 1.0
    assert np.isnan(points2d_mask[1, 3, 0])
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


def test_auto_threshold_selector():
    selector_config = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/auto_threshold_selector.py'))
    selector = build_point_selector(selector_config)
    # test input not changed
    points2d = np.ones((3, 17, 3))
    points2d[:, 0, 2] = np.array((0.4, 0.6, 0.6))
    points2d[:, 1, 2] = np.array((0.7, 0.6, 0.6))
    points2d[:, 2, 2] = np.array((0.7, 0.6, 0.3))
    points2d[:, 3, 2] = np.array((0.8, 0.8, 0.8))
    init_mask = np.ones((3, 17, 1))
    # keypoints 3 does not anticipate triangulation
    init_mask[:, 3, :] = np.nan
    points2d_backup = points2d.copy()
    init_mask_backup = init_mask.copy()
    points2d_mask = selector.get_selection_mask(points2d, init_mask)
    assert np.all(points2d_backup == points2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    # test selection results
    assert np.all(points2d_mask[:, 0, 0] == np.array((0, 1, 1)))
    assert np.all(points2d_mask[:, 1, 0] == np.array((1, 1, 1)))
    assert np.all(points2d_mask[:, 2, 0] == np.array((1, 1, 0)))
    assert np.allclose(
        points2d_mask[:, 3, 0], np.array((np.nan, ) * 3), equal_nan=True)
    # test no potential
    init_mask[0, 0, 0] = 0
    init_mask_backup = init_mask.copy()
    points2d_mask = selector.get_selection_mask(points2d, init_mask)
    assert np.all(points2d_backup == points2d)
    assert np.allclose(init_mask_backup, init_mask, equal_nan=True)
    # test selection results
    assert np.all(points2d_mask[:, 0, 0] == np.array((0, 1, 1)))
    assert np.all(points2d_mask[:, 1, 0] == np.array((1, 1, 1)))
    assert np.all(points2d_mask[:, 2, 0] == np.array((1, 1, 0)))
    assert np.allclose(
        points2d_mask[:, 3, 0], np.array((np.nan, ) * 3), equal_nan=True)
