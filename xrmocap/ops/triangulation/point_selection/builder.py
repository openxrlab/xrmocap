from mmcv.utils import Registry

from .auto_threshold_selector import AutoThresholdSelector
from .camera_error_selector import CameraErrorSelector
from .manual_threshold_selector import ManualThresholdSelector
from .slow_camera_error_selector import SlowCameraErrorSelector

POINTSELECTORS = Registry('point_selector')

POINTSELECTORS.register_module(
    name='AutoThresholdSelector', module=AutoThresholdSelector)
POINTSELECTORS.register_module(
    name='ManualThresholdSelector', module=ManualThresholdSelector)
POINTSELECTORS.register_module(
    name='SlowCameraErrorSelector', module=SlowCameraErrorSelector)
POINTSELECTORS.register_module(
    name='CameraErrorSelector', module=CameraErrorSelector)


def build_point_selector(cfg):
    """Build point selector."""
    return POINTSELECTORS.build(cfg)
