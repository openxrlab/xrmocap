# yapf: disable
from mmcv.utils import Registry

from .base_data_visualization import BaseDataVisualization
from .mview_mperson_data_visualization import MviewMpersonDataVisualization

# yapf: enable

DATA_VISUALIZATION = Registry('data_visualization')
DATA_VISUALIZATION.register_module(
    name='MviewMpersonDataVisualization', module=MviewMpersonDataVisualization)


def build_data_visualization(cfg) -> BaseDataVisualization:
    """Build data_visualization."""
    return DATA_VISUALIZATION.build(cfg)
