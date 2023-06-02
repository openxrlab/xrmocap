# yapf: disable
from mmcv.utils import Registry

from .base_data_converter import BaseDataCovnerter
from .campus_data_converter import CampusDataCovnerter
from .humman_smc_data_converter import HummanSMCDataCovnerter
from .panoptic_data_converter import PanopticDataCovnerter
from .shelf_data_converter import ShelfDataCovnerter

# yapf: enable

DATA_CONVERTERS = Registry('data_converter')
DATA_CONVERTERS.register_module(
    name='CampusDataCovnerter', module=CampusDataCovnerter)
DATA_CONVERTERS.register_module(
    name='ShelfDataCovnerter', module=ShelfDataCovnerter)
DATA_CONVERTERS.register_module(
    name='PanopticDataCovnerter', module=PanopticDataCovnerter)
DATA_CONVERTERS.register_module(
    name='HummanSMCDataCovnerter', module=HummanSMCDataCovnerter)


def build_data_converter(cfg) -> BaseDataCovnerter:
    """Build data_converter."""
    return DATA_CONVERTERS.build(cfg)
