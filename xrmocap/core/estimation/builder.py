# yapf: disable
from mmcv.utils import Registry

from .base_estimator import BaseEstimator
from .mview_mperson_smpl_estimator import MultiViewMultiPersonSMPLEstimator
from .mview_sperson_smpl_estimator import MultiViewSinglePersonSMPLEstimator

# yapf: enable

ESTIMATORS = Registry('estimator')
ESTIMATORS.register_module(
    name='MultiViewSinglePersonSMPLEstimator',
    module=MultiViewSinglePersonSMPLEstimator)
ESTIMATORS.register_module(
    name='MultiViewMultiPersonSMPLEstimator',
    module=MultiViewMultiPersonSMPLEstimator)


def build_estimator(cfg) -> BaseEstimator:
    """Build estimator."""
    return ESTIMATORS.build(cfg)
