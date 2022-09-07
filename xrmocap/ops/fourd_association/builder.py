# yapf: disable
from mmcv.utils import Registry

from .fourd_associator import FourdAssociator
# yapf: enable

TOP_DOWN_ASSOCIATORS = Registry('fourd_associator')

TOP_DOWN_ASSOCIATORS.register_module(
    name='FourdAssociator', module=FourdAssociator)

def build_fourd_associator(cfg) -> FourdAssociator:
    """Build top_down_associator."""
    return TOP_DOWN_ASSOCIATORS.build(cfg)
