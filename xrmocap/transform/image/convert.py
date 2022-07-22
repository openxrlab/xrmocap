import logging
import numpy as np
from PIL import Image
from typing import Union

from .base_image_transform import BaseImageTransform


class CV2ToPIL(BaseImageTransform):

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Convert cv2 image array to PIL.Image.

        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseImageTransform.__init__(self, logger=logger)

    def forward(self, input: np.ndarray) -> Image:
        """Forward function of CV2ToPIL.

        Args:
            input (np.ndarray):
                Image array defined in cv2.

        Returns:
            Image:
                Image instance defined in PIL.
        """
        pil_img = Image.fromarray(input)
        return pil_img
