import cv2
import logging
import numpy as np
from PIL import Image
from typing import Union

from .base_image_transform import BaseImageTransform


class LoadImageCV2(BaseImageTransform):

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Load image array from file by cv2.

        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseImageTransform.__init__(self, logger=logger)

    def forward(self, input: str) -> np.ndarray:
        """Forward function of LoadImageCV2.

        Args:
            input (str):
                Path to the image file.

        Returns:
            np.ndarray:
                Image instance defined in PIL.
        """
        img = cv2.imread(input)
        return img


class LoadImagePIL(BaseImageTransform):

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Load image array from file by PIL.

        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseImageTransform.__init__(self, logger=logger)

    def forward(self, input: str) -> Image:
        """Forward function of LoadImagePIL.

        Args:
            input (str):
                Path to the image file.

        Returns:
            Image:
                Image instance defined in PIL.
        """
        img = Image.open(input)
        return img
