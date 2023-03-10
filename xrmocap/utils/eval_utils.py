# yapf: disable
import logging
import numpy as np
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import (
    convert_keypoints, get_intersection_mask, get_keypoint_idx,
)
from xrmocap.utils.mvpose_utils import (
    add_campus_jaw_headtop, add_campus_jaw_headtop_mask,
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
    gt_nose = None
    pred_nose = None
    pred_kps3d_convention = pred_keypoints3d_raw.get_convention()
    gt_kps3d_convention = gt_keypoints3d_raw.get_convention()
    if gt_kps3d_convention == 'panoptic':
        gt_nose_index = get_keypoint_idx(
            name='nose_openpose', convention=gt_kps3d_convention)
        gt_nose = gt_keypoints3d_raw.get_keypoints()[:, :, gt_nose_index, :3]

    if pred_kps3d_convention == 'coco':
        pred_nose_index = get_keypoint_idx(
            name='nose', convention=pred_kps3d_convention)
        pred_nose = pred_keypoints3d_raw.get_keypoints()[:, :,
                                                         pred_nose_index, :3]

    if pred_kps3d_convention == 'fourdag_19' or\
            pred_kps3d_convention == 'openpose_25':
        pred_leftear_index = get_keypoint_idx(
            name='left_ear_openpose', convention=pred_kps3d_convention)
        pre_rightear_index = get_keypoint_idx(
            name='right_ear_openpose', convention=pred_kps3d_convention)
        head_center = (
            pred_keypoints3d_raw.get_keypoints()[:, :, pred_leftear_index, :3]
            + pred_keypoints3d_raw.get_keypoints()[:, :,
                                                   pre_rightear_index, :3]) / 2
        pred_nose = head_center

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

    pred_kps3d_mask = pred_keypoints3d.get_mask()
    pred_kps3d = pred_keypoints3d.get_keypoints()[..., :3]
    if pred_nose is not None and eval_kps3d_convention == 'campus':
        pred_kps3d = add_campus_jaw_headtop(pred_nose, pred_kps3d)
        pred_kps3d_mask = add_campus_jaw_headtop_mask(pred_kps3d_mask)

    gt_kps3d_mask = gt_keypoints3d.get_mask()
    gt_kps3d = gt_keypoints3d.get_keypoints()[..., :3]
    if gt_nose is not None and eval_kps3d_convention == 'campus':
        gt_kps3d = add_campus_jaw_headtop(gt_nose, gt_kps3d)
        gt_kps3d_mask = add_campus_jaw_headtop_mask(gt_kps3d_mask)

    pred_kps3d = np.concatenate((pred_kps3d, pred_kps3d_mask[..., np.newaxis]),
                                axis=-1)
    pred_keypoints3d = Keypoints(
        kps=pred_kps3d, mask=pred_kps3d_mask, convention=eval_kps3d_convention)
    gt_kps3d = np.concatenate((gt_kps3d, gt_kps3d_mask[..., np.newaxis]),
                              axis=-1)
    gt_keypoints3d = Keypoints(
        kps=gt_kps3d, mask=gt_kps3d_mask, convention=eval_kps3d_convention)

    if pred_kps3d_convention != gt_kps3d_convention:
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
