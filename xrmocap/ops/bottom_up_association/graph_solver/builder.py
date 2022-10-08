from mmcv.utils import Registry

from .graph_associate import GraphAssociate
from .graph_construct import GraphConstruct

GRAPHSOLVER = Registry('grapph_solver')

GRAPHSOLVER.register_module(name='GraphAssociate', module=GraphAssociate)
GRAPHSOLVER.register_module(name='GraphConstruct', module=GraphConstruct)


def build_graph_solver(cfg):
    """Build a graph solver instance."""
    return GRAPHSOLVER.build(cfg)
