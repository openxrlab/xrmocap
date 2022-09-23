from mmcv.utils import Registry

from .fourdag_associate import FourDAGAssociate

ASSOCIATE = Registry('associate')

ASSOCIATE.register_module(name='FourDAGAssociate', module=FourDAGAssociate)


def build_associate(cfg) -> FourDAGAssociate:
    """Build a associate instance."""
    return ASSOCIATE.build(cfg)
