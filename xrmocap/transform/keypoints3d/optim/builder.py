from mmcv.utils import Registry

from .base_optimizer import BaseOptimizer
from .nan_interpolation import NanInterpolation

KEYPOINTS3D_OPTIMIZERS = Registry('keypoints3d_optimizer')

KEYPOINTS3D_OPTIMIZERS.register_module(
    name='NanInterpolation', module=NanInterpolation)


def build_keypoints3d_optimizer(cfg) -> BaseOptimizer:
    """Build keypoints3d optimizer."""
    return KEYPOINTS3D_OPTIMIZERS.build(cfg)
