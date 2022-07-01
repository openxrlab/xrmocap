from mmcv.utils import Registry

from .smplify import SMPLify
from .smplifyx import SMPLifyX
from .smplifyxd import SMPLifyXD

REGISTRANTS = Registry('registrant')
REGISTRANTS.register_module(name='SMPLify', module=SMPLify)
REGISTRANTS.register_module(name='SMPLifyX', module=SMPLifyX)
REGISTRANTS.register_module(name='SMPLifyXD', module=SMPLifyXD)


def build_registrant(cfg) -> SMPLify:
    """Build registrant."""
    return REGISTRANTS.build(cfg)
