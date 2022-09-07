# yapf: disable
from mmcv.utils import Registry

from .base_estimator import BaseEstimator
from .mperson_smpl_estimator import MultiPersonSMPLEstimator
from .mview_sperson_smpl_estimator import MultiViewSinglePersonSMPLEstimator

# yapf: enable

ESTIMATORS = Registry('estimator')
ESTIMATORS.register_module(
    name='MultiViewSinglePersonSMPLEstimator',
    module=MultiViewSinglePersonSMPLEstimator)
ESTIMATORS.register_module(
    name='MultiPersonSMPLEstimator', module=MultiPersonSMPLEstimator)


def build_estimator(cfg) -> BaseEstimator:
    """Build estimator."""
    return ESTIMATORS.build(cfg)
