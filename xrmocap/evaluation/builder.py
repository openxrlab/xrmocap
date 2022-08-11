from mmcv.utils import Registry

from .top_down_association_evaluation import TopDownAssociationEvaluation

EVALUATION = Registry('evaluation')

EVALUATION.register_module(
    name='TopDownAssociationEvaluation', module=TopDownAssociationEvaluation)


def build_evaluation(cfg) -> TopDownAssociationEvaluation:
    """Build a matching instance."""
    return EVALUATION.build(cfg)
