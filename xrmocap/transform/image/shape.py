import cv2
import logging
import numpy as np
import torch
from typing import Tuple, Union

from xrmocap.utils.geometry import get_affine_transform, get_scale
from .base_image_transform import BaseImageTransform


class WarpAffine(BaseImageTransform):

    def __init__(self,
                 image_size: np.ndarray,
                 flag: str = 'inter_linear',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Warp image array with given affine transformation matrix and given
        image size.

        Args:
            image_size (np.ndarray):
                Output image size.
            flag (str, optional):
                Interpolation method for resizing. Defaults to 'inter_linear'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseImageTransform.__init__(self, logger=logger)
        flags_dict = {
            'inter_linear': cv2.INTER_LINEAR,
            'inter_nearest': cv2.INTER_NEAREST,
            'inter_area': cv2.INTER_AREA,
            'inter_cubid': cv2.INTER_CUBIC,
            'inter_lanczos4': cv2.INTER_LANCZOS4
        }

        if flag in flags_dict.keys():
            self.flag = flags_dict[flag]
        else:
            self.logger.warning(
                'Unrecognized flag, using cv2.INTER_LINEAR by default.')
            self.flag = cv2.INTER_LINEAR

        self.image_size = image_size

    def forward(self, input: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Forward function of WarpAffine.

        Args:
            input (np.ndarray):
                An array of image with [h, w, n_ch].

        Returns:
            np.ndarray
        """
        height, width = input.shape[0:2]
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0  # NOTE: do not apply rotation augmentation
        trans = get_affine_transform(c, s, r, self.image_size, inv=0)
        warped_img = cv2.warpAffine(
            input,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=self.flag)
        return warped_img


def get_affine_trans_aug(
    c: np.ndarray,
    s: np.ndarray,
    aug_s: np.ndarray,
    image_size: np.ndarray,
    r: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get affine transformation matrix, inverse affine transformation matrix
    and augmented affine transformation with given.

    Args:
        c (np.ndarray): Center of the image.
        s (np.ndarray): Scale for affine transformation.
        aug_s (np.ndarray): Scale for augmented affine transformation.
        image_size (np.ndarray): Original image size.
        r (int, optional): Rotation. Defaults to 0.

    Returns:
        aff_trans (np.ndarray): Affine transformation matrix.
        inv_aff_trans (np.ndarray): Inverse affine transformation matrix.
        aug_trans (np.ndarray): Augmented affine transformation matrix.
    """
    aff_trans = np.eye(3, 3)
    inv_aff_trans = np.eye(3, 3)
    aug_trans = np.eye(3, 3)
    scale_trans = np.eye(3, 3)

    trans = get_affine_transform(c, s, r, image_size, inv=0)
    inv_trans = get_affine_transform(c, s, r, image_size, inv=1)

    aff_trans[0:2] = trans.copy()
    inv_aff_trans[0:2] = inv_trans.copy()
    aug_trans[0:2] = trans.copy()

    scale_trans[0, 0] = aug_s[1]
    scale_trans[1, 1] = aug_s[0]
    aug_trans = scale_trans.dot(aug_trans)

    return aff_trans, inv_aff_trans, aug_trans
