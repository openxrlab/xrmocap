# yapf: disable
import numpy
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_rotation_6d,
)
from typing import Union

from xrmocap.transform.convention.joints_convention.standard_joint_angles import (  # noqa:E501
    TRANSFORMATION_AA_TO_SJA, TRANSFORMATION_SJA_TO_AA,
)

# yapf: enable


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, numpy.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, numpy.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_sja(
    axis_angle: Union[torch.Tensor, numpy.ndarray],
    R_t: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_AA_TO_SJA,
    R_t_inv: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_SJA_TO_AA
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis-angles to standard joint angles.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3), ndim of input is unlimited.
        R_t (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                original axis-angle coordinate system to
                standard joint angle coordinate system,
                ndim of input is unlimited.
        R_t_inv (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                standard joint angle coordinate system to
                original axis-angle coordinate system,
                ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """

    def _aa_to_sja(aa, R_t, R_t_inv):
        R_aa = axis_angle_to_matrix(aa)
        R_sja = R_t @ R_aa @ R_t_inv
        sja = matrix_to_euler_angles(R_sja, convention='XYZ')
        return sja

    if axis_angle.shape[-2:] != (21, 3):
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    if R_t.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t shape f{R_t.shape}.')
    if R_t_inv.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t_inv shape f{R_t.shape}.')
    t = Compose([_aa_to_sja])
    return t(axis_angle, R_t=R_t, R_t_inv=R_t_inv)


def aa_to_rot6d(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)
