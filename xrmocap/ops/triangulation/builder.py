from mmcv.utils import Registry
from xrprimer.ops.triangulation.builder import TRIANGULATORS  # noqa:F401
from xrprimer.ops.triangulation.builder import build_triangulator  # noqa:F401

POINTSELECTORS = Registry('point_selector')


def build_point_selector(cfg):
    """Build point selector."""
    return POINTSELECTORS.build(cfg)
