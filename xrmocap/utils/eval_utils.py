# yapf: disable
import logging
import numpy as np
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_intersection_mask, get_keypoint_idx,
)

# yapf: enable


def align_by_keypoint(keypoints: Keypoints, keypoint_name='right_ankle'):
    convention = keypoints.get_convention()
    index = get_keypoint_idx(name=keypoint_name, convention=convention)
    if index == -1:
        raise ValueError('Check the convention of kps3d!')

    kps = keypoints.get_keypoints()
    aligned_kps = np.zeros_like(kps)
    n_frame, n_person = kps.shape[:2]
    for frame_idx in range(n_frame):
        for person_idx in range(n_person):
            aligned_kps[frame_idx, person_idx, ...] = \
                kps[frame_idx, person_idx, :] - \
                kps[frame_idx, person_idx, index, :]
    return aligned_kps


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


def align_convention_mask(pred_keypoints3d_raw: Keypoints,
                          gt_keypoints3d_raw: Keypoints,
                          pred_kps3d_convention: str,
                          gt_kps3d_convention: str,
                          eval_kps3d_convention: str,
                          logger: Union[None, str, logging.Logger] = None):
    """Convert pred and gt to the same convention before passing to metric
    manager, human_data as eval convention is recommended. Set mask to
    intersection mask if pred and gt convention are in different convention.

    Args:
        pred_keypoints3d_raw (Keypoints):
            Predicted 3D keypoints in original convention.
        gt_keypoints3d_raw (Keypoints):
            Ground-truth 3D keypoints in original convention.
        pred_kps3d_convention (str):
            Original convention of predicted 3D keypoints.
        gt_kps3d_convention (str):
            Original convention of ground-truth 3D keypoints.
        eval_kps3d_convention (str):
            Convention used for alignment and evaluation.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be
            selected. Defaults to None.

    Returns:
        Keypoints:
            Aligned predicted 3D keyopints and
            ground-truth 3D keypoints
    """

    logger = get_logger(logger)
    if pred_kps3d_convention != eval_kps3d_convention:
        pred_keypoints3d = convert_keypoints(
            pred_keypoints3d_raw, dst=eval_kps3d_convention, approximate=True)
    else:
        pred_keypoints3d = pred_keypoints3d_raw

    if gt_kps3d_convention != eval_kps3d_convention:
        gt_keypoints3d = convert_keypoints(
            gt_keypoints3d_raw, dst=eval_kps3d_convention, approximate=True)
    else:
        gt_keypoints3d = gt_keypoints3d_raw

    if pred_kps3d_convention != gt_kps3d_convention:
        if eval_kps3d_convention != 'human_data':
            logger.warning(
                'Predicion and ground truth is having'
                'different convention. It is recommended to set '
                'eval_kps3d_convention to human_data to avoid error.')

        intersection_mask = get_intersection_mask(pred_kps3d_convention,
                                                  gt_kps3d_convention,
                                                  eval_kps3d_convention)
        gt_intersection_mask = np.multiply(gt_keypoints3d.get_mask(),
                                           intersection_mask)
        pred_intersection_mask = np.multiply(pred_keypoints3d.get_mask(),
                                             intersection_mask)
        gt_keypoints3d.set_mask(gt_intersection_mask)
        pred_keypoints3d.set_mask(pred_intersection_mask)

    return gt_keypoints3d, pred_keypoints3d
