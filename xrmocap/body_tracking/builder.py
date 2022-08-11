# yapf: disable
from mmcv.utils import Registry

from .kalman_tracking import KalmanTracking

# yapf: enable

KALMAN_TRACKING = Registry('kalman_tracking')
KALMAN_TRACKING.register_module(name='KalmanTracking', module=KalmanTracking)


def build_kalman_tracking(cfg) -> KalmanTracking:
    """Build kalman_tracking."""
    return KALMAN_TRACKING.build(cfg)
