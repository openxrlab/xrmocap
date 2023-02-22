from mmcv.utils import Registry

from .base_metric import BaseMetric
from .mpjpe_metric import MPJPEMetric
from .pa_mpjpe_metric import PAMPJPEMetric
from .pck_metric import PCKMetric
from .pcp_metric import PCPMetric
from .precision_recall_metric import PrecisionRecallMetric
from .prediction_matcher import PredictionMatcher

METRICS = Registry('metrics')

METRICS.register_module(name='PredictionMatcher', module=PredictionMatcher)
METRICS.register_module(name='MPJPEMetric', module=MPJPEMetric)
METRICS.register_module(name='PAMPJPEMetric', module=PAMPJPEMetric)
METRICS.register_module(name='PCKMetric', module=PCKMetric)
METRICS.register_module(name='PCPMetric', module=PCPMetric)
METRICS.register_module(
    name='PrecisionRecallMetric', module=PrecisionRecallMetric)


def build_metric(cfg) -> BaseMetric:
    """Build an evaluation metric."""
    return METRICS.build(cfg)
