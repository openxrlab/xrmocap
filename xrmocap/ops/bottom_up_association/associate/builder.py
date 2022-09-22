from mmcv.utils import Registry

from .base_associate import BaseAssociate
from .fourdag_associate import FourDAGAssociate

ASSOCIATE = Registry('associate')

ASSOCIATE.register_module(name='FourDAGAssociate', module=FourDAGAssociate)


def build_associate(cfg) -> BaseAssociate:
    """Build a associate instance."""
    return ASSOCIATE.build(cfg)
