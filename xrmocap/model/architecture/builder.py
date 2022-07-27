from mmcv.utils import Registry

from .affinity_estimator import AppearanceAffinityEstimator
from .multi_view_pose_transformer import MviewPoseTransformer

ARCHITECTURES = Registry('architectures')

ARCHITECTURES.register_module(
    name='AppearanceAffinityEstimator', module=AppearanceAffinityEstimator)
ARCHITECTURES.register_module(
    name='MviewPoseTransformer', module=MviewPoseTransformer)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
