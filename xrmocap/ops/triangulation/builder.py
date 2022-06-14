from mmcv.utils import Registry

from xrprimer.ops.triangulation.builder import (  # noqa:F401
    TRIANGULATORS, BaseTriangulator, build_triangulator,
)
from .aniposelib_triangulator import AniposelibTriangulator
from .point_selection.auto_threshold_selector import AutoThresholdSelector
from .point_selection.base_selector import BaseSelector
from .point_selection.camera_error_selector import CameraErrorSelector
from .point_selection.manual_threshold_selector import ManualThresholdSelector
from .point_selection.slow_camera_error_selector import SlowCameraErrorSelector

TRIANGULATORS.register_module(
    name='AniposelibTriangulator', module=AniposelibTriangulator)

POINTSELECTORS = Registry('point_selector')

POINTSELECTORS.register_module(
    name='AutoThresholdSelector', module=AutoThresholdSelector)
POINTSELECTORS.register_module(
    name='ManualThresholdSelector', module=ManualThresholdSelector)
POINTSELECTORS.register_module(
    name='SlowCameraErrorSelector', module=SlowCameraErrorSelector)
POINTSELECTORS.register_module(
    name='CameraErrorSelector', module=CameraErrorSelector)


def build_point_selector(cfg) -> BaseSelector:
    """Build point selector."""
    return POINTSELECTORS.build(cfg)
