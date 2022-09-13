from mmcv.utils import Registry

from .auto_threshold_selector import AutoThresholdSelector
from .base_selector import BaseSelector
from .camera_error_selector import CameraErrorSelector
from .hybrid_kps2d_selector import HybridKps2dSelector
from .manual_threshold_selector import ManualThresholdSelector
from .reprojection_error_point_selector import ReprojectionErrorPointSelector
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
POINTSELECTORS.register_module(
    name='HybridKps2dSelector', module=HybridKps2dSelector)
POINTSELECTORS.register_module(
    name='ReprojectionErrorPointSelector',
    module=ReprojectionErrorPointSelector)


def build_point_selector(cfg) -> BaseSelector:
    """Build point selector."""
    return POINTSELECTORS.build(cfg)
