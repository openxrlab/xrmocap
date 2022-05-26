from mmcv.utils import Registry

from .smpl import SMPL

BODYMODELS = Registry('body_model')

BODYMODELS.register_module(name='SMPL', module=SMPL)


def build_body_model(cfg):
    """Build body model."""
    return BODYMODELS.build(cfg)
