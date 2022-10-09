from mmcv.utils import Registry

from .bottom_up_association_evaluation import BottomUpAssociationEvaluation
from .mvp_evaluation import MVPEvaluation
from .top_down_association_evaluation import TopDownAssociationEvaluation

EVALUATION = Registry('evaluation')

EVALUATION.register_module(
    name='TopDownAssociationEvaluation', module=TopDownAssociationEvaluation)
EVALUATION.register_module(name='MVPEvaluation', module=MVPEvaluation)
EVALUATION.register_module(
    name='BottomUpAssociationEvaluation', module=BottomUpAssociationEvaluation)


def build_evaluation(cfg):
    """Build a matching instance."""
    return EVALUATION.build(cfg)
