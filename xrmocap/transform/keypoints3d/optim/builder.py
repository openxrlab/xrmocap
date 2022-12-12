from mmcv.utils import Registry

from .aniposelib_optimizer import AniposelibOptimizer
from .base_optimizer import BaseOptimizer
from .fourdag_base_optimizer import FourDAGBaseOptimizer
from .fourdag_optimization import FourDAGOptimizer
from .median_smooth import MedianSmooth
from .nan_interpolation import NanInterpolation
from .rm_duplicate import RemoveDuplicate
from .smpl_shape_aware_optimizer import SMPLShapeAwareOptimizer
from .trajectory_optimizer import TrajectoryOptimizer

KEYPOINTS3D_OPTIMIZERS = Registry('keypoints3d_optimizer')

KEYPOINTS3D_OPTIMIZERS.register_module(
    name='NanInterpolation', module=NanInterpolation)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='TrajectoryOptimizer', module=TrajectoryOptimizer)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='MedianSmooth', module=MedianSmooth)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='AniposelibOptimizer', module=AniposelibOptimizer)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='SMPLShapeAwareOptimizer', module=SMPLShapeAwareOptimizer)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='RemoveDuplicate', module=RemoveDuplicate)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='FourDAGBaseOptimizer', module=FourDAGBaseOptimizer)
KEYPOINTS3D_OPTIMIZERS.register_module(
    name='FourDAGOptimizer', module=FourDAGOptimizer)


def build_keypoints3d_optimizer(cfg) -> BaseOptimizer:
    """Build keypoints3d optimizer."""
    return KEYPOINTS3D_OPTIMIZERS.build(cfg)
