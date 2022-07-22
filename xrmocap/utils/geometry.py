import logging
import numpy as np
import torch
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.convention.bbox_convention import convert_bbox


def compute_similarity_transform(X: np.ndarray,
                                 Y: np.ndarray,
                                 compute_optimal_scale=False):
    """A port of MATLAB's `procrustes` function to Numpy. Adapted from
    http://stackoverflow.com/a/18927641/1884420.

    Args:
        X (np.ndarray): Array NxM of targets, with N number of points and
           M point dimensionality.
        Y (np.ndarray): Array NxM of inputs
        compute_optimal_scale (bool, optional): whether we compute optimal
            scale or force it to be 1. Defaults to False.

    Returns:
        d: squared error after transformation
        Z (np.ndarray): transformed Y
        T (np.ndarray): computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c


def compute_iou(
    rec_1: Union[np.ndarray, torch.Tensor],
    rec_2: Union[np.ndarray, torch.Tensor],
    bbox_convention: str = 'xyxy',
    logger: Union[None, str, logging.Logger] = None
) -> Union[np.float64, torch.Tensor]:
    """Compute iou.

    Args:
        rec_1 (Union[np.ndarray, torch.Tensor]):
            Bbox2d data. The last dim should be
            (x, y, x, y, score), or (x, y, w, h, score), and score is optional.
        rec_2 (Union[np.ndarray, torch.Tensor]):
            Bbox2d data. The last dim should be
            (x, y, x, y, score), or (x, y, w, h, score), and score is optional.
        bbox_convention (str, optional): Bbox type, xyxy or xywh.
            Defaults to 'xyxy'.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.
            (rec_1)
        1--------1
        1   1----1------1
        1---1----1      1
            1           1
            1-----------1 (rec_2)
    Returns:
        Union[np.float64, torch.Tensor]: iou
    """
    logger = get_logger(logger)
    assert bbox_convention == 'xywh' or bbox_convention == 'xyxy'
    if bbox_convention == 'xywh':
        rec_1 = convert_bbox(rec_1, src='xywh', dst='xyxy')
        rec_2 = convert_bbox(rec_2, src='xywh', dst='xyxy')

    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])
    sum_s = s_rec1 + s_rec2
    left = max(rec_1[0], rec_2[0])
    right = min(rec_1[2], rec_2[2])
    bottom = max(rec_1[1], rec_2[1])
    top = min(rec_1[3], rec_2[3])
    if left >= right or top <= bottom:
        if isinstance(rec_1, torch.Tensor):
            return torch.tensor(0)
        else:
            return np.float64(0)
    else:
        inter = (right - left) * (top - bottom)
        iou = (inter / (sum_s - inter)) * 1.0
        return iou
