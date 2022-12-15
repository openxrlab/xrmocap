# yapf: disable
from mmcv.utils import Registry

from .bbox_detection.mmdet_detector import MMdetDetector
from .bbox_detection.mmtrack_detector import MMtrackDetector
from .keypoints_estimation.mediapipe_estimator import MediapipeEstimator
from .keypoints_estimation.mmpose_top_down_estimator import (
    MMposeTopDownEstimator,
)

# yapf: enable

DETECTORS = Registry('detector')
DETECTORS.register_module(
    name=('MMposeTopDownEstimator'), module=MMposeTopDownEstimator)
DETECTORS.register_module(
    name=('MediapipeEstimator'), module=MediapipeEstimator)
DETECTORS.register_module(name=('MMdetDetector'), module=MMdetDetector)
DETECTORS.register_module(name=('MMtrackDetector'), module=MMtrackDetector)


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)
