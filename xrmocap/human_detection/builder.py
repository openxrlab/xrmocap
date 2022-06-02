# yapf: disable
from mmcv.utils import Registry

from .bbox_detection.mmdet_detector import MMdetDetector
from .keypoints_estimation.mmpose_top_down_estimator import (
    MMposeTopDownEstimator,
)

# yapf: enable

DETECTORS = Registry('detector')
DETECTORS.register_module(
    name=('MMposeTopDownEstimator'), module=MMposeTopDownEstimator)
DETECTORS.register_module(name=('MMdetDetector'), module=MMdetDetector)


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)
