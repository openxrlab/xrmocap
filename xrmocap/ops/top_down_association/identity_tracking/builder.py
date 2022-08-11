from mmcv.utils import Registry

from .base_tracking import BaseTracking
from .keypoints_distance_tracking import KeypointsDistanceTracking
from .perception2d_tracking import Perception2dTracking

IDENTITY_TRACKINGS = Registry('identity_tracking')

IDENTITY_TRACKINGS.register_module(
    name='Perception2dTracking', module=Perception2dTracking)
IDENTITY_TRACKINGS.register_module(
    name='KeypointsDistanceTracking', module=KeypointsDistanceTracking)


def build_identity_tracking(cfg) -> BaseTracking:
    """Build a identity tracking class."""
    return IDENTITY_TRACKINGS.build(cfg)
