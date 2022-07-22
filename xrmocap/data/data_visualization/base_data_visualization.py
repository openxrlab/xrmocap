import logging
from typing import Union
from xrprimer.utils.log_utils import get_logger
from xrprimer.utils.path_utils import prepare_output_path


class BaseDataVisualization:

    def __init__(self,
                 data_root: str,
                 output_dir: str,
                 meta_path: str = 'xrmocap_meta',
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class of all data visualizations.

        Args:
            data_root (str):
                Root path of the downloaded dataset.
            output_dir (str):
                Path to the output dir.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            verbose (bool, optional):
                Whether to print(logger.info) information.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.data_root = data_root
        self.meta_path = meta_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.logger = get_logger(logger)

    def run(self, overwrite: bool = False) -> None:
        """Convert data from original dataset to meta-data defined by xrmocap.

        Args:
            overwrite (bool, optional):
                Whether replace the files at
                self.output_dir.
                Defaults to False.
        """
        prepare_output_path(
            output_path=self.output_dir,
            allowed_suffix='',
            path_type='dir',
            overwrite=overwrite,
            logger=self.logger)
