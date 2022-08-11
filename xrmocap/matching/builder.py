from mmcv.utils import Registry

from .base_matching import BaseMatching
from .multi_way_matching import MultiWayMatching

MATCHING = Registry('matching')

MATCHING.register_module(name='MultiWayMatching', module=MultiWayMatching)


def build_matching(cfg) -> BaseMatching:
    """Build a matching instance."""
    return MATCHING.build(cfg)
