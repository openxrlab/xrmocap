from mmcv.utils import Registry

from .smplify import SMPLify

REGISTRANTS = Registry('registrant')
REGISTRANTS.register_module(name='SMPLify', module=SMPLify)


def build_registrant(cfg):
    """Build registrant."""
    return REGISTRANTS.build(cfg)
