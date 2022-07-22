import logging
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.utils.path_utils import prepare_output_path


class BaseDataCovnerter:

    def __init__(self,
                 data_root: str,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'base_dataset',
                 verbose: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class of all data converters. It create a dir at meta_path,

        , and put dataset_name and meta-data into it.

        Args:
            data_root (str):
                Root path of the downloaded dataset.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset. Defaults to 'base_dataset'.
            verbose (bool, optional):
                Whether to print(logger.info) information.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.data_root = data_root
        self.meta_path = meta_path
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.logger = get_logger(logger)

    def run(self, overwrite: bool = False) -> None:
        """Convert data from original dataset to meta-data defined by xrmocap.

        Args:
            overwrite (bool, optional):
                Whether replace the files at
                self.meta_path.
                Defaults to False.
        """
        prepare_output_path(
            output_path=self.meta_path,
            allowed_suffix='',
            path_type='dir',
            overwrite=overwrite,
            logger=self.logger)
