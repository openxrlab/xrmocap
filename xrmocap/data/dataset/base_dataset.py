import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.image.builder import build_image_transform


class BaseDataset(Dataset):

    def __init__(self,
                 data_root: str,
                 img_pipeline: list,
                 meta_path: str = 'xrmocap_meta',
                 dataset_name: str = 'base_dataset',
                 test_mode: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Base class of all Dataset in XRMocap. It loads data from source
        dataset and meta-data from data converter.

        Args:
            data_root (str):
                Root path of the downloaded dataset.
            img_pipeline (list):
                A list of image transform instances.
            meta_path (str, optional):
                Path to the meta-data dir. Defaults to 'xrmocap_meta'.
            dataset_name (str, optional):
                Name of the dataset. Defaults to 'base_dataset'.
            test_mode (bool, optional):
                Whether this dataset is used to load testset.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__()
        self.data_root = data_root
        self.meta_path = meta_path
        self.dataset_name = dataset_name
        self.test_mode = test_mode
        self.logger = get_logger(logger)
        self.img_pipeline = []
        for transform in img_pipeline:
            if isinstance(transform, dict):
                transform = build_image_transform(transform)
            self.img_pipeline.append(transform)
        self.img_pipeline = Compose(self.img_pipeline)
