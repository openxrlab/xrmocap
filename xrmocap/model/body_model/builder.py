from mmcv.utils import Registry

from .smpl import SMPL
from .smplx import SMPLX

BODYMODELS = Registry('body_model')

BODYMODELS.register_module(name='SMPL', module=SMPL)
BODYMODELS.register_module(name='SMPLX', module=SMPLX)


def build_body_model(cfg):
    """Build body model."""
    return BODYMODELS.build(cfg)
