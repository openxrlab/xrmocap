from mmcv.utils import Registry

from .smplify_base_hook import SMPLifyBaseHook
from .smplify_verbose_hook import SMPLifyVerboseHook

SMPLIFY_HOOKS = Registry('smplify_hook')
SMPLIFY_HOOKS.register_module(
    name='SMPLifyVerboseHook', module=SMPLifyVerboseHook)


def build_smplify_hook(cfg) -> SMPLifyBaseHook:
    """Build a hook for smplify."""
    return SMPLIFY_HOOKS.build(cfg)
