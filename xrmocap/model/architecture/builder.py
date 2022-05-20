from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from .affinity_estimator import AppearanceAffinityEstimator


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


ARCHITECTURES = Registry('architectures', build_func=build_from_cfg)

ARCHITECTURES.register_module(
    name='AppearanceAffinityEstimator', module=AppearanceAffinityEstimator)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
