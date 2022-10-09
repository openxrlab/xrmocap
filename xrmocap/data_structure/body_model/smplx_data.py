# yapf: disable

import logging
import numpy as np
import torch
from typing import Any, Union

from .smpl_data import SMPLData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# yapf: enable


class SMPLXData(SMPLData):
    BODY_POSE_LEN = 21
    HAND_POSE_LEN = 15
    JAW_POSE_LEN = 1
    EYE_POSE_LEN = 1
    DEFAULT_BODY_JOINTS_NUM = 144
    BODY_POSE_KEYS = {
        'global_orient',
        'body_pose',
    }
    FULL_POSE_KEYS = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose'
    }

    def __init__(self,
                 src_dict: dict = None,
                 gender: Union[Literal['female', 'male', 'neutral'],
                               None] = None,
                 fullpose: Union[np.ndarray, torch.Tensor, None] = None,
                 transl: Union[np.ndarray, torch.Tensor, None] = None,
                 betas: Union[np.ndarray, torch.Tensor, None] = None,
                 expression: Union[np.ndarray, torch.Tensor, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a SMPLXData instance with pre-set values. If any of
        gender, fullpose, transl, betas is provided, it will override the item
        in source_dict.

        Args:
            src_dict (dict, optional):
                A dict with items in HumanData fashion.
                Defaults to None.
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
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        SMPLData.__init__(
            self,
            src_dict=src_dict,
            gender=gender,
            transl=transl,
            fullpose=fullpose,
            betas=betas,
            logger=logger)
        if expression is None and 'expression' not in self:
            expression = np.zeros(shape=(self.get_batch_size(), 10))
        if expression is not None:
            self.set_expression(expression)
        self.body_joints_num = self.__class__.DEFAULT_BODY_JOINTS_NUM

    @classmethod
    def get_fullpose_dim(cls) -> int:
        """Get dimension of full pose.

        Returns:
            int:
                Dim value. Full pose shall be
                in shape (frame_n, dim, 3)
        """
        global_orient_dim = 1
        ret_sum = global_orient_dim + cls.BODY_POSE_LEN + \
            cls.JAW_POSE_LEN + 2 * cls.EYE_POSE_LEN + \
            2 * cls.HAND_POSE_LEN
        return ret_sum

    def set_expression(self, expression: Union[np.ndarray,
                                               torch.Tensor]) -> None:
        """Set expression data.

        Args:
            expression (Union[np.ndarray, torch.Tensor]):
                Expression parameters in ndarray or tensor,
                in shape [batch_size, n].
                n stands for any positive int, typically it's 10.

        Raises:
            TypeError: Type of expression is not correct.
        """
        if isinstance(expression, torch.Tensor):
            expression = expression.detach().cpu().numpy()
        elif not isinstance(expression, np.ndarray):
            self.logger.error('Type of expression is not correct.\n' +
                              f'Type: {type(expression)}.')
            raise TypeError
        if len(expression.shape) == 1:
            expression = expression[np.newaxis, ...]
        expression_dim = expression.shape[-1]
        expression_np = expression.reshape(-1, expression_dim)
        dict.__setitem__(self, 'expression', expression_np)

    def get_expression(self, repeat_expression: bool = True) -> np.ndarray:
        """Get expression.

        Args:
            repeat_expression (bool, optional):
                Whether to repeat expression when its first dim doesn't match
                batch_size. Defaults to True.

        Returns:
            ndarray:
                expression in shape [batch_size, expression_dims] or
                [1, expression_dims].
        """
        batch_size = self.get_global_orient().shape[0]
        expression = self.__getitem__('expression')
        if repeat_expression and\
                expression.shape[0] == 1 and\
                expression.shape[0] != batch_size:
            expression = expression.repeat(repeats=batch_size, axis=0)
        return expression

    def __setitem__(self, __k: Any, __v: Any) -> None:
        """Set item according to its key.

        Args:
            __k (Any): Key in dict.
            __v (Any): Value in dict.
        """
        if __k == 'expression':
            self.set_expression(__v)
        else:
            SMPLData.__setitem__(self, __k, __v)

    def from_param_dict(self, smplx_dict: dict) -> None:
        """Load SMPLX parameters from smplx_dict.

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
        necessary_keys = {'global_orient', 'body_pose'}
        if not necessary_keys.issubset(smplx_dict):
            self.logger.error('Keys are not enough.\n' +
                              f'smplx_dict\'s keys: {smplx_dict.keys()}')
            raise KeyError
        global_orient = smplx_dict['global_orient']
        batch_size = global_orient.shape[0] \
            if len(global_orient.shape) > 1 else 1
        global_orient = smplx_dict['global_orient'].reshape(batch_size, 3)
        body_pose = smplx_dict['body_pose'].reshape(batch_size, -1)
        if isinstance(global_orient, torch.Tensor):

            def concat_func(data_list, dim):
                return torch.cat(data_list, dim=dim)

            def zeros_func(shape, ref_data):
                return torch.zeros(
                    size=shape, dtype=ref_data.dtype, device=ref_data.device)

        elif isinstance(global_orient, np.ndarray):

            def concat_func(data_list, dim):
                return np.concatenate(data_list, axis=dim)

            def zeros_func(shape, ref_data):
                return np.zeros(shape=shape, dtype=ref_data.dtype)

        jaw_pose = smplx_dict['jaw_pose'].reshape(batch_size, -1) \
            if 'jaw_pose' in smplx_dict else\
            zeros_func(
                shape=[batch_size, self.__class__.JAW_POSE_LEN*3],
                ref_data=global_orient)
        leye_pose = smplx_dict['leye_pose'].reshape(batch_size, -1) \
            if 'leye_pose' in smplx_dict else\
            zeros_func(
                shape=[batch_size, self.__class__.EYE_POSE_LEN*3],
                ref_data=global_orient)
        reye_pose = smplx_dict['reye_pose'].reshape(batch_size, -1) \
            if 'reye_pose' in smplx_dict else\
            zeros_func(
                shape=[batch_size, self.__class__.EYE_POSE_LEN*3],
                ref_data=global_orient)
        if 'left_hand_pose' in smplx_dict:
            if smplx_dict['left_hand_pose'].reshape(
                    batch_size, -1, 3).shape[1] == \
                        self.__class__.HAND_POSE_LEN:
                left_hand_pose = smplx_dict['left_hand_pose'].reshape(
                    batch_size, -1)
            else:
                left_hand_pose = zeros_func(
                    shape=[batch_size, self.__class__.HAND_POSE_LEN * 3],
                    ref_data=global_orient)
                self.logger.warning(
                    'SMPLX is using pca for hands,' +
                    ' left_hand_pose in SMPLXData will be set to zeros.')
        else:
            left_hand_pose = zeros_func(
                shape=[batch_size, self.__class__.HAND_POSE_LEN * 3],
                ref_data=global_orient)
        if 'right_hand_pose' in smplx_dict:
            if smplx_dict['right_hand_pose'].reshape(
                    batch_size, -1, 3).shape[1] == \
                        self.__class__.HAND_POSE_LEN:
                right_hand_pose = smplx_dict['right_hand_pose'].reshape(
                    batch_size, -1)
            else:
                right_hand_pose = zeros_func(
                    shape=[batch_size, self.__class__.HAND_POSE_LEN * 3],
                    ref_data=global_orient)
                self.logger.warning(
                    'SMPLX is using pca for hands,' +
                    ' right_hand_pose in SMPLXData will be set to zeros.')
        else:
            right_hand_pose = zeros_func(
                shape=[batch_size, self.__class__.HAND_POSE_LEN * 3],
                ref_data=global_orient)

        fullpose = concat_func([
            global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
            left_hand_pose, right_hand_pose
        ],
                               dim=1)
        self.set_fullpose(fullpose)
        if 'transl' in smplx_dict:
            self.set_transl(smplx_dict['transl'])
        if 'betas' in smplx_dict:
            self.set_betas(smplx_dict['betas'])

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
        dict_to_return = SMPLData.to_param_dict(
            self, repeat_betas=repeat_betas)
        dict_to_return.pop('body_pose')
        fullpose = self.get_fullpose()
        start_idx = 1
        body_pose = fullpose[:, start_idx:start_idx +
                             self.__class__.BODY_POSE_LEN].reshape(
                                 self.get_batch_size(), -1)
        start_idx += self.__class__.BODY_POSE_LEN
        jaw_pose = fullpose[:, start_idx:start_idx +
                            self.__class__.JAW_POSE_LEN].reshape(-1, 3)
        start_idx += self.__class__.JAW_POSE_LEN
        leye_pose = fullpose[:, start_idx:start_idx +
                             self.__class__.EYE_POSE_LEN].reshape(-1, 3)
        start_idx += self.__class__.EYE_POSE_LEN
        reye_pose = fullpose[:, start_idx:start_idx +
                             self.__class__.EYE_POSE_LEN].reshape(-1, 3)
        start_idx += self.__class__.EYE_POSE_LEN
        left_hand_pose = fullpose[:, start_idx:start_idx +
                                  self.__class__.HAND_POSE_LEN].reshape(-1, 3)
        start_idx += self.__class__.HAND_POSE_LEN
        right_hand_pose = fullpose[:, start_idx:start_idx +
                                   self.__class__.HAND_POSE_LEN].reshape(
                                       -1, 3)
        expression = self.get_expression(repeat_expression=repeat_expression)
        dict_to_return.update({
            'body_pose': body_pose,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'expression': expression
        })
        return dict_to_return

    def to_tensor_dict(self,
                       repeat_betas: bool = True,
                       repeat_expression: bool = True,
                       device: Union[torch.device, str] = 'cpu') -> dict:
        """It is almost same as self.to_param_dict, but all the values are
        tensors in a specified device. Split fullpose into global_orient and
        body_pose, return all the necessary parameters in one dict.

        Args:
            repeat_betas (bool, optional):
                Whether to repeat betas when its first dim doesn't match
                batch_size. Defaults to True.
            repeat_expression (bool, optional):
                Whether to repeat expression when its first dim doesn't match
                batch_size. Defaults to True.
            device (Union[torch.device, str], optional):
                A specified device. Defaults to CPU_DEVICE. Defaults to 'cpu'.

        Returns:
            dict: A dict of SMPLX data, whose keys are
                betas, body_pose, global_orient and transl, etc.
        """
        np_dict = self.to_param_dict(
            repeat_betas=repeat_betas, repeat_expression=repeat_expression)
        dict_to_return = {}
        for key, value in np_dict.items():
            dict_to_return[key] = torch.tensor(
                value, device=device, dtype=torch.float32)
        return dict_to_return
