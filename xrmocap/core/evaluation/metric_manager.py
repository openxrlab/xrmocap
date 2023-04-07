# yapf: disable
import logging
from typing import List, Union
from xrprimer.utils.log_utils import get_logger

from .metrics.base_metric import BaseMetric
from .metrics.builder import build_metric

# yapf: enable


class MetricManager:
    """MetricManager is a class for multiple metrics evaluation.

    It sorts metrics by ranks(ascend), and call the calculation functions in
    turn.
    """

    def __init__(
        self,
        metric_list: List[Union[dict, BaseMetric]],
        pick_dict: Union[dict, None] = None,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        """Construction of MetricManager.

        Args:
            metric_list (List[Union[dict, BaseMetric]]):
                A list of metric instances or configs.
            pick_dict (Union[dict, None], optional):
                Defines which keys to pick for every metric.
                Defaults to None, pick all.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        # build metrics
        unsorted_list = []
        for metric in metric_list:
            if isinstance(metric, dict):
                metric['logger'] = self.logger
                metric = build_metric(metric)
            unsorted_list.append(metric)
        # sort metrics by RANK
        self.metric_list = _sort_metrics_by_rank(unsorted_list)
        # set pick dict
        self.set_pick_dict(pick_dict)

    def set_pick_dict(self, raw_pick_dict: Union[dict, None]):
        """Set which keys to pick for every metric.

        Args:
            raw_pick_dict (Union[dict, None]):
                Defines which keys to pick for every metric.
                If None, pick all.
        """
        if raw_pick_dict is None:
            pick_dict = dict()
            for metric in self.metric_list:
                name = metric.name
                pick_value = 'all'
                pick_dict[name] = pick_value
        else:
            pick_dict = raw_pick_dict
            for key in list(pick_dict.keys()):
                value = pick_dict[key]
                if value != 'all' and \
                        not isinstance(value, list):
                    pick_dict[key] = [
                        value,
                    ]
        self.pick_dict = pick_dict

    def __call__(self, *args, **kwargs) -> dict:
        """Calculate metrics one by one, and return the picked contents.

        Returns:
            dict: picked results.
        """
        accumulate_kwargs = dict()
        accumulate_kwargs.update(kwargs)
        manager_ret_dict = dict()
        for metric in self.metric_list:
            name = metric.name
            # calculate metric
            metric_ret_dict = metric(*args, **accumulate_kwargs)
            # update accumulate_kwargs for the next metric
            accumulate_kwargs.update(metric_ret_dict)
            # record return keys and values
            if name in self.pick_dict:
                selections = self.pick_dict[name]
                if selections == 'all':
                    manager_ret_dict[name] = metric_ret_dict
                else:
                    manager_ret_dict[name] = dict()
                    for key in selections:
                        manager_ret_dict[name][key] = metric_ret_dict[key]
        return manager_ret_dict, accumulate_kwargs


def _sort_metrics_by_rank(unsorted_list: List[BaseMetric]) -> List[BaseMetric]:
    sorted_list = []
    cur_rank = 0
    unsorted_idxs = list(range(len(unsorted_list)))
    while len(unsorted_idxs) > 0:
        new_unsorted_idxs = []
        for idx in unsorted_idxs:
            metric = unsorted_list[idx]
            rank = metric.__class__.RANK
            # rank matches, add to sorted list
            if rank == cur_rank:
                sorted_list.append(metric)
            # not matched yet, preprare for next iter
            else:
                new_unsorted_idxs.append(idx)
        unsorted_idxs = new_unsorted_idxs
        cur_rank += 1
    return sorted_list
