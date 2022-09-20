
from .fourdag_optimization import FourDAGOptimization
from .base_optimization import BaseOptimization

from mmcv.utils import Registry


PARAMETRIC_OPTIMIZATION = Registry('parametric_optimization')

PARAMETRIC_OPTIMIZATION.register_module(
    name='FourDAGOptimization', module=FourDAGOptimization)
PARAMETRIC_OPTIMIZATION.register_module(name='BaseOptimization', module=BaseOptimization)


def build_parametric_optimization(cfg) -> BaseOptimization:
    return PARAMETRIC_OPTIMIZATION.build(cfg)