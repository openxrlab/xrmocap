from mmcv.utils import Registry

from .base_matching import BaseMatching
from .fourd_matching import FourdMatching

MATCHING = Registry('matching')

MATCHING.register_module(name='FourdMatching', module=FourdMatching)


def build_matching(cfg) -> BaseMatching:
    """Build a matching instance."""
    return MATCHING.build(cfg)
