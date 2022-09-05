# yapf: disable
from mmcv.utils import Registry

from .base_estimator import BaseEstimator
from .mmse import MMSE
from .mperson_smpl_estimator import MultiPersonSMPLEstimator
from .mview_sperson_smpl_estimator import MultiViewSinglePersonSMPLEstimator

# yapf: enable

ESTIMATORS = Registry('estimator')
ESTIMATORS.register_module(
    name='MultiViewSinglePersonSMPLEstimator',
    module=MultiViewSinglePersonSMPLEstimator)
ESTIMATORS.register_module(
    name='MultiPersonSMPLEstimator', module=MultiPersonSMPLEstimator)
ESTIMATORS.register_module(name='MMSE', module=MMSE)


def build_estimator(cfg) -> BaseEstimator:
    """Build estimator."""
    return ESTIMATORS.build(cfg)
