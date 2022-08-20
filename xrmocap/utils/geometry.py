import cv2
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


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """This function calculates affine transformation from given center, scale,
    rotation and shift."""

    def get_3rd_point(a, b):
        # get the third point from the first two points to define
        # an affine transformation
        direct = a - b
        return np.array(b) + np.array([-direct[1], direct[0]],
                                      dtype=np.float32)

    if isinstance(scale, torch.Tensor):
        scale = np.array(scale.cpu())
    if isinstance(center, torch.Tensor):
        center = np.array(center.cpu())
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = output_size[0], output_size[1]

    rot_rad = np.pi * rot / 180
    if src_w >= src_h:
        src_dir = get_direction([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
    else:
        src_dir = get_direction([src_h * -0.5, 0], rot_rad)
        dst_dir = np.array([dst_h * -0.5, 0], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift  # x,y
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_direction(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_scale(image_size, resized_size):
    w, h = image_size
    w_resized, h_resized = resized_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized
    scale = np.array([w_pad / 200.0, h_pad / 200.0], dtype=np.float32)

    return scale


def project_3dpts(X, K, R, t, Kd):
    """Projects points X (3xN) using camera intrinsics K (3x3), extrinsics
    (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/
    calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = torch.mm(R, X) + t
    # x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = (
        x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) +
        2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (r + 2 * x[0, :] * x[0, :]))
    x[1, :] = (
        x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) +
        2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (r + 2 * x[1, :] * x[1, :]))

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


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
