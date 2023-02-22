# yapf: disable
import logging
import os
from typing import List, Union
from xrprimer.utils.log_utils import get_logger
from xrprimer.utils.path_utils import prepare_output_path

from xrmocap.data.data_visualization.builder import (
    BaseDataVisualization, build_data_visualization,
)
from xrmocap.data.dataset.builder import MviewMpersonDataset, build_dataset
from .metric_manager import MetricManager
from .metrics.base_metric import BaseMetric

# yapf: enable


class BaseEvaluation:

    def __init__(
        self,
        dataset: Union[dict, MviewMpersonDataset],
        output_dir: str,
        metric_list: List[Union[dict, BaseMetric]],
        pick_dict: Union[dict, None] = None,
        dataset_visualization: Union[None, dict, BaseDataVisualization] = None,
        eval_kps3d_convention: str = 'human_data',
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        self.logger = get_logger(logger)
        self.output_dir = output_dir
        self.eval_kps3d_convention = eval_kps3d_convention
        self.metric_manager = MetricManager(
            metric_list=metric_list, pick_dict=pick_dict, logger=self.logger)

        if isinstance(dataset, dict):
            dataset['logger'] = self.logger
            self.dataset = build_dataset(dataset)
        else:
            self.dataset = dataset

        if isinstance(dataset_visualization, dict):
            dataset_visualization['logger'] = self.logger
            self.dataset_visualization = build_data_visualization(
                dataset_visualization)
        else:
            self.dataset_visualization = dataset_visualization

    def run(self, overwrite: bool = False):
        if not os.path.exists(self.output_dir):
            prepare_output_path(
                output_path=self.output_dir,
                allowed_suffix='',
                path_type='dir',
                overwrite=overwrite,
                logger=self.logger)
