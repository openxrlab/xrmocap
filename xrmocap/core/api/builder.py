# yapf: disable
from mmcv.utils import Registry

from .base_api import BaseAPI
from .mview_sperson_smpl_estimator import MultiViewSinglePersonSMPLEstimator

# yapf: enable

APIS = Registry('api')
APIS.register_module(
    name='MultiViewSinglePersonSMPLEstimator',
    module=MultiViewSinglePersonSMPLEstimator)


def build_api(cfg) -> BaseAPI:
    """Build api."""
    return APIS.build(cfg)
