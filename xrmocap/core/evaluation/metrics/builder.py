from mmcv.utils import Registry

from .base_metric import BaseMetric
from .mpjpe_metric import MPJPEMetric
from .pa_mpjpe_metric import PAMPJPEMetric
from .pck_metric import PCKMetric

METRICS = Registry('metrics')

METRICS.register_module(name='MPJPEMetric', module=MPJPEMetric)
METRICS.register_module(name='PAMPJPEMetric', module=PAMPJPEMetric)
METRICS.register_module(name='PCKMetric', module=PCKMetric)


def build_metric(cfg) -> BaseMetric:
    """Build an evaluation metric."""
    return METRICS.build(cfg)
