# yapf: disable
from mmcv.utils import Registry

from .bottom_up_associator import BottomUpAssociator
# yapf: enable

BOTTOM_UP_ASSOCIATORS = Registry('fourd_associator')

BOTTOM_UP_ASSOCIATORS.register_module(
    name='BottomUpAssociator', module=BottomUpAssociator)

def build_bottom_up_associator(cfg) -> BottomUpAssociator:
    """Build top_down_associator."""
    return BOTTOM_UP_ASSOCIATORS.build(cfg)
