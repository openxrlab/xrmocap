from mmcv.utils import Registry

from .aniposelib_optimizer import AniposelibOptimizer
from .base_optimizer import BaseOptimizer
from .median_smooth import MedianSmooth
from .nan_interpolation import NanInterpolation

KEYPOINTS3D_OPTIMIZERS = Registry('keypoints3d_optimizer')

KEYPOINTS3D_OPTIMIZERS.register_module(
    name='NanInterpolation', module=NanInterpolation)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='MedianSmooth', module=MedianSmooth)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='AniposelibOptimizer', module=AniposelibOptimizer)


def build_keypoints3d_optimizer(cfg) -> BaseOptimizer:
    """Build keypoints3d optimizer."""
    return KEYPOINTS3D_OPTIMIZERS.build(cfg)
