from mmcv.utils import Registry

from .bottom_up_association_evaluation import BottomUpAssociationEvaluation
from .end2end_evaluation import End2EndEvaluation
from .top_down_association_evaluation import TopDownAssociationEvaluation

EVALUATION = Registry('evaluation')

EVALUATION.register_module(
    name='TopDownAssociationEvaluation', module=TopDownAssociationEvaluation)
EVALUATION.register_module(name='End2EndEvaluation', module=End2EndEvaluation)
EVALUATION.register_module(
    name='BottomUpAssociationEvaluation', module=BottomUpAssociationEvaluation)


def build_evaluation(cfg):
    """Build a matching instance."""
    return EVALUATION.build(cfg)
