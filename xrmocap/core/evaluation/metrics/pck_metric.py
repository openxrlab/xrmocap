# yapf: disable
import logging
import numpy as np
from typing import Union

from .base_metric import BaseMetric

# yapf: enable


class PCKMetric(BaseMetric):
    RANK = 0

    def __init__(
        self,
        name: str,
        threshold: float,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        BaseMetric.__init__(self, name=name, logger=logger)
        self.threshold = threshold

    def __call__(self, pa_mpjpe_value: float, **kwargs):
        pck_value = np.mean(pa_mpjpe_value <= self.threshold)
        return dict(pck_value=pck_value)
