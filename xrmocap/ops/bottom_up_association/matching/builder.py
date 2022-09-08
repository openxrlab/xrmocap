from mmcv.utils import Registry

from .base_matching import BaseMatching
from .fourdag_matching import FourDAGMatching

MATCHING = Registry('matching')

MATCHING.register_module(name='FourDAGMatching', module=FourDAGMatching)


def build_matching(cfg) -> BaseMatching:
    """Build a matching instance."""
    return MATCHING.build(cfg)
