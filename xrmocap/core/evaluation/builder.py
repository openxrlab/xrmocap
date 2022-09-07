from mmcv.utils import Registry

from .mvp_evaluation import MVPEvaluation
from .top_down_association_evaluation import TopDownAssociationEvaluation
from .fourd_association_evaluation import FourDAssociationEvaluation
EVALUATION = Registry('evaluation')

EVALUATION.register_module(
    name='TopDownAssociationEvaluation', module=TopDownAssociationEvaluation)
EVALUATION.register_module(name='MVPEvaluation', module=MVPEvaluation)
EVALUATION.register_module(
    name='FourDAssociationEvaluation', module=FourDAssociationEvaluation)


def build_evaluation(cfg):
    """Build a matching instance."""
    return EVALUATION.build(cfg)
