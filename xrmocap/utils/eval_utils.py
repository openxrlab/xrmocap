import numpy as np

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import get_keypoint_idx


def align_by_keypoint(keypoints: Keypoints, keypoint_name='right_ankle'):
    convention = keypoints.get_convention()
    index = get_keypoint_idx(name=keypoint_name, convention=convention)
    if index == -1:
        raise ValueError('Check the convention of kps3d!')
    # kps = keypoints.get_keypoints()[0, 0, :, :3]
    # root = kps[index, :]
    # return kps - root
    
    kps = keypoints.get_keypoints()
    aligned_kps = np.zeros_like(kps)
    n_frame, n_person = kps.shape[:2]
    for frame_idx in range(n_frame):
        for person_idx in range(n_person):
            aligned_kps[frame_idx, person_idx, ...] = \
                kps[frame_idx, person_idx, :] - kps[frame_idx, person_idx, index, :]
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
