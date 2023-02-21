import numpy as np
import torch

from xrmocap.transform.image.color import bgr2rgb, rgb2bgr


def test_bgr2rgb():
    # test numpy
    # one img
    rgb_image = np.zeros(shape=(3, 1920, 1080))
    rgb_image[2, ...] = 2
    assert rgb_image[2, 0, 0] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=0)
    assert bgr_image[0, 0, 0] == 2
    assert bgr_image[2, 0, 0] == 0
    bgr_image = rgb2bgr(rgb_image, color_dim=0)
    assert bgr_image[0, 0, 0] == 2
    assert bgr_image[2, 0, 0] == 0
    # pytorch batch like
    rgb_image = np.zeros(shape=(2, 3, 1920, 1080))
    rgb_image[:, 2, ...] = 2
    assert rgb_image[0, 2, 0, 0] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=1)
    assert bgr_image[0, 0, 0, 0] == 2
    assert bgr_image[0, 2, 0, 0] == 0
    # opencv video like
    rgb_image = np.zeros(shape=(2, 1920, 1080, 3))
    rgb_image[..., 2] = 2
    assert rgb_image[0, 0, 0, 2] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=-1)
    assert bgr_image[0, 0, 0, 0] == 2
    assert bgr_image[0, 0, 0, 2] == 0
    # test torch
    # one img
    rgb_image = torch.zeros(size=(3, 1920, 1080))
    rgb_image[2, ...] = 2
    assert rgb_image[2, 0, 0] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=0)
    assert bgr_image[0, 0, 0] == 2
    assert bgr_image[2, 0, 0] == 0
    # pytorch batch like
    rgb_image = torch.zeros(size=(2, 3, 1920, 1080))
    rgb_image[:, 2, ...] = 2
    assert rgb_image[0, 2, 0, 0] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=1)
    assert bgr_image[0, 0, 0, 0] == 2
    assert bgr_image[0, 2, 0, 0] == 0
    # opencv video like
    rgb_image = torch.zeros(size=(2, 1920, 1080, 3))
    rgb_image[..., 2] = 2
    assert rgb_image[0, 0, 0, 2] == 2
    bgr_image = bgr2rgb(rgb_image, color_dim=-1)
    assert bgr_image[0, 0, 0, 0] == 2
    assert bgr_image[0, 0, 0, 2] == 0
    # test in-place
    rgb_image = np.zeros(shape=(3, 1920, 1080))
    rgb_image[2, ...] = 2
    rgb2bgr(rgb_image, color_dim=0, inplace=True)
    assert rgb_image[0, 0, 0] == 2
    rgb_image = torch.zeros(size=(3, 1920, 1080))
    rgb_image[2, ...] = 2
    rgb2bgr(rgb_image, color_dim=0, inplace=True)
    assert rgb_image[0, 0, 0] == 2
