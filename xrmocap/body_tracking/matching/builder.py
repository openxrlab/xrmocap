from mmcv.utils import Registry

from .base_matching import BaseMatching
from .dfs_homo_matching import DFSHomoMatching
from .dfs_matching import DFSMatching

MATCHING = Registry('matching')

MATCHING.register_module(name='DFSMatching', module=DFSMatching)
MATCHING.register_module(name='DFSHomoMatching', module=DFSHomoMatching)


def build_matching(cfg) -> BaseMatching:
    """Build a matching instance."""
    return MATCHING.build(cfg)
