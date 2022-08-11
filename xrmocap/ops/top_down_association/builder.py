# yapf: disable
from mmcv.utils import Registry

from .mvpose_associator import MvposeAssociator

# yapf: enable

TOP_DOWN_ASSOCIATORS = Registry('top_down_associator')
TOP_DOWN_ASSOCIATORS.register_module(
    name='MvposeAssociator', module=MvposeAssociator)


def build_top_down_associator(cfg) -> MvposeAssociator:
    """Build top_down_associator."""
    return TOP_DOWN_ASSOCIATORS.build(cfg)
