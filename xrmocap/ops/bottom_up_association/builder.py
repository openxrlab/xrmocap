# yapf: disable
from mmcv.utils import Registry

from .fourdag_associator import FourDAGAssociator
# yapf: enable

BOTTOM_UP_ASSOCIATORS = Registry('bottom_up_associator')

BOTTOM_UP_ASSOCIATORS.register_module(
    name='FourDAGAssociator', module=FourDAGAssociator)

def build_bottom_up_associator(cfg) -> FourDAGAssociator:
    """Build top_down_associator."""
    return BOTTOM_UP_ASSOCIATORS.build(cfg)
