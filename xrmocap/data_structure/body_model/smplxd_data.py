# yapf: disable

import logging
import numpy as np
import torch
from typing import Any, Union

from .smplx_data import SMPLXData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# yapf: enable


class SMPLXDData(SMPLXData):
    NUM_VERTS = 10475

    def __init__(self,
                 gender: Union[Literal['female', 'male', 'neutral'],
                               None] = None,
                 fullpose: Union[np.ndarray, torch.Tensor, None] = None,
                 transl: Union[np.ndarray, torch.Tensor, None] = None,
                 betas: Union[np.ndarray, torch.Tensor, None] = None,
                 expression: Union[np.ndarray, torch.Tensor, None] = None,
                 displacement: Union[np.ndarray, torch.Tensor, None] = None,
                 mask: Union[np.ndarray, torch.Tensor, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a SMPLXData instance with pre-set values.

        Args:
            gender (Union[
                    Literal['female', 'male', 'neutral'], None], optional):
                Gender of the body model.
                Should be one among ["female", "male", "neutral"].
                Defaults to None.
            fullpose (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for fullpose, in shape [frame_num, 24, 3].
                Defaults to None, zero-tensor will be created.
            transl (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for translation, in shape [frame_num, 3].
                Defaults to None, zero-tensor will be created.
            betas (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for betas,
                in shape [frame_num, betas_dim].
                Defaults to None,
                zero-tensor in shape [frame_num, 10] will be created.
            expression (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for expression,
                in shape [frame_num, expression_dim].
                Defaults to None,
                zero-tensor in shape [frame_num, 10] will be created.
            displacement (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for displacement,
                in shape [frame_num, NUM_VERTS, 3].
                Defaults to None,
                zero-tensor in shape [frame_num, NUM_VERTS] will be created.
            mask (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for framewise visibility mask,
                in shape [n_frame, ].
                Defaults to None,
                one-tensor in shape [n_frame, ] will be created.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        SMPLXData.__init__(
            self,
            gender=gender,
            transl=transl,
            fullpose=fullpose,
            betas=betas,
            expression=expression,
            mask=mask,
            logger=logger)
        if displacement is None and 'displacement' not in self:
            displacement = np.zeros(
                shape=(self.get_batch_size(), self.__class__.NUM_VERTS, 3))
        if displacement is not None:
            self.set_displacement(displacement)

    @classmethod
    def from_dict(cls, smpl_data_dict: Union['SMPLXDData',
                                             dict]) -> 'SMPLXDData':
        """Construct a body model data structure from a SMPLXDData, or a
        degraded smplxd_data in dict type.

        Args:
            smplxd_data_dict (dict):
                A degraded smplxd_data in dict type.

        Returns:
            SMPLXData:
                A SMPLXDData instance load from dict.
        """
        smplxd_data_dict = smpl_data_dict
        min_keys = {
            'gender', 'fullpose', 'transl', 'betas', 'expression',
            'displacement'
        }
        assert min_keys <= smplxd_data_dict.keys()
        ret_instance = cls(
            gender=smplxd_data_dict['gender'],
            fullpose=smplxd_data_dict['fullpose'],
            transl=smplxd_data_dict['transl'],
            betas=smplxd_data_dict['betas'],
            expression=smplxd_data_dict['expression'],
            displacement=smplxd_data_dict['displacement'])
        return ret_instance

    def set_displacement(
            self, displacement: Union[np.ndarray, torch.Tensor]) -> None:
        """Set displacement data.

        Args:
            displacement (Union[np.ndarray, torch.Tensor]):
                Displacement parameters in ndarray or tensor,
                in shape [batch_size, NUM_VERTS, 3].

        Raises:
            TypeError: Type of displacement is not correct.
        """
        if isinstance(displacement, torch.Tensor):
            displacement = displacement.detach().cpu().numpy()
        elif not isinstance(displacement, np.ndarray):
            self.logger.error('Type of displacement is not correct.\n' +
                              f'Type: {type(displacement)}.')
            raise TypeError
        if len(displacement.shape) < 3:
            self.logger.error('Shape of displacement is not correct.\n' +
                              f'Shape: {type(displacement.shape)}.')
            raise ValueError
        dict.__setitem__(self, 'displacement', displacement)

    def get_displacement(self) -> np.ndarray:
        """Get displacement.

        Returns:
            ndarray:
                Displacement in shape [batch_size, NUM_VERTS, 3].
        """
        displacement = self.__getitem__('displacement')
        return displacement

    def __setitem__(self, __k: Any, __v: Any) -> None:
        """Set item according to its key.

        Args:
            __k (Any): Key in dict.
            __v (Any): Value in dict.
        """
        if __k == 'displacement':
            self.set_displacement(__v)
        else:
            SMPLXData.__setitem__(self, __k, __v)

    def from_param_dict(self, smplxd_dict: dict) -> None:
        """Load SMPLX+D parameters from smplxd_dict, which is the output of a
        body model in most cases.

        Args:
            smplx_dict (dict):
                A dict of ndarray|Tensor parameters.
                global_orient and body_pose are necessary,
                jaw_pose, leye_pose, reye_pose,
                left_hand_pose, right_hand_pose, transl and
                betas are optional.
                Other keys are ignored.

        Raises:
            KeyError: missing necessary keys.
        """
        necessary_keys = {'global_orient', 'body_pose', 'displacement'}
        if not necessary_keys.issubset(smplxd_dict):
            self.logger.error('Keys are not enough.\n' +
                              f'smplxd_dict\'s keys: {smplxd_dict.keys()}')
            raise KeyError
        SMPLXData.from_param_dict(self, smplxd_dict)
        self.set_displacement(smplxd_dict['displacement'])

    def to_param_dict(self,
                      repeat_betas: bool = True,
                      repeat_expression: bool = True) -> dict:
        """Split fullpose into global_orient, body_pose, jaw_pose, leye_pose,
        reye_pose, left_hand_pose, right_hand_pose, return all the necessary
        parameters in one dict.

        Args:
            repeat_betas (bool, optional):
                Whether to repeat betas when its first dim doesn't match
                batch_size. Defaults to True.
            repeat_expression (bool, optional):
                Whether to repeat expression when its first dim doesn't match
                batch_size. Defaults to True.

        Returns:
            dict:
                A dict of SMPLX data, whose keys are
                betas, global_orient, transl, global_orient, body_pose,
                jaw_pose, leye_pose, reye_pose, left_hand_pose,
                right_hand_pose, expression.
        """
        dict_to_return = SMPLXData.to_param_dict(
            self,
            repeat_betas=repeat_betas,
            repeat_expression=repeat_expression)
        dict_to_return['displacement'] = self.get_displacement()
        return dict_to_return
