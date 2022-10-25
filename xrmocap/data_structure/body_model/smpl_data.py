# yapf: disable
import logging
import numpy as np
import torch
from typing import Any, Union
from xrprimer.utils.log_utils import get_logger
from xrprimer.utils.path_utils import (
    Existence, check_path_existence, check_path_suffix,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# yapf: enable


class SMPLData(dict):
    BODY_POSE_LEN = 23
    DEFAULT_BODY_JOINTS_NUM = 45
    BODY_POSE_KEYS = {
        'global_orient',
        'body_pose',
    }
    FULL_POSE_KEYS = {
        'global_orient',
        'body_pose',
    }

    def __init__(self,
                 src_dict: dict = None,
                 gender: Union[Literal['female', 'male', 'neutral'],
                               None] = None,
                 fullpose: Union[np.ndarray, torch.Tensor, None] = None,
                 transl: Union[np.ndarray, torch.Tensor, None] = None,
                 betas: Union[np.ndarray, torch.Tensor, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a SMPLData instance with pre-set values. If any of gender,
        fullpose, transl, betas is provided, it will override the item in
        source_dict.

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
                A tensor or ndarray for fullpose, in shape [n_frame, 24, 3].
                Defaults to None, zero-tensor will be created.
            transl (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for translation, in shape [n_frame, 3].
                Defaults to None, zero-tensor will be created.
            betas (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for translation,
                in shape [n_frame, betas_dim].
                Defaults to None,
                zero-tensor in shape [n_frame, 10] will be created.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        if src_dict is not None:
            super().__init__(src_dict)
        else:
            super().__init__()
        self.n_body_joints = self.__class__.DEFAULT_BODY_JOINTS_NUM
        self.logger = get_logger(logger)
        if gender is None and 'gender' not in self:
            gender = 'neutral'
        if gender is not None:
            self.set_gender(gender)
        if fullpose is None and 'fullpose' not in self:
            fullpose_dim = self.__class__.get_fullpose_dim()
            fullpose = np.zeros(shape=(1, fullpose_dim, 3))
        if fullpose is not None:
            self.set_fullpose(fullpose)
        if transl is None and 'transl' not in self:
            transl = np.zeros(shape=(self.get_batch_size(), 3))
        if transl is not None:
            self.set_transl(transl)
        if betas is None and 'betas' not in self:
            betas = np.zeros(shape=(self.get_batch_size(), 10))
        if betas is not None:
            self.set_betas(betas)

    @classmethod
    def fromfile(cls, npz_path: str) -> 'SMPLData':
        """Construct a body model data structure from an npz file.

        Args:
            npz_path (str):
                Path to a dumped npz file.

        Returns:
            SMPLData:
                A SMPLData instance load from file.
        """
        ret_instance = cls()
        ret_instance.load(npz_path)
        return ret_instance

    @classmethod
    def get_fullpose_dim(cls) -> int:
        """Get dimension of full pose.

        Returns:
            int:
                Dim value. Full pose shall be
                in shape (frame_n, dim, 3)
        """
        global_orient_dim = 1
        ret_sum = global_orient_dim + cls.BODY_POSE_LEN
        return ret_sum

    def set_gender(
            self,
            gender: Literal['female', 'male', 'neutral'] = 'neutral') -> None:
        """Set gender.

        Args:
            gender (Literal["female", "male", "neutral"], optional):
                Gender of the body model.
                Should be one among ["female", "male", "neutral"].
                Defaults to 'neutral'.

        Raises:
            ValueError: Value of gender is not correct.
        """
        if gender in ['female', 'male', 'neutral']:
            super().__setitem__('gender', gender)
        else:
            self.logger.error('Value of gender is not correct.\n' +
                              f'gender: {gender}.')
            raise ValueError

    def set_fullpose(self, fullpose: Union[np.ndarray, torch.Tensor]) -> None:
        """Set full pose data.

        Args:
            fullpose (Union[np.ndarray, torch.Tensor]):
                Full pose in ndarray or tensor,
                in shape [batch_size, fullpose_dim, 3].
                global_orient at [:, 0, :].

        Raises:
            TypeError: Type of fullpose is not correct.
        """
        fullpose_dim = self.__class__.get_fullpose_dim()
        if isinstance(fullpose, np.ndarray):
            fullpose_np = fullpose.reshape(-1, fullpose_dim, 3)
        elif isinstance(fullpose, torch.Tensor):
            fullpose_np = fullpose.detach().cpu().numpy().reshape(
                -1, fullpose_dim, 3)
        else:
            self.logger.error('Type of fullpose is not correct.\n' +
                              f'Type: {type(fullpose)}.')
            raise TypeError
        super().__setitem__('fullpose', fullpose_np)

    def set_transl(self, transl: Union[np.ndarray, torch.Tensor]) -> None:
        """Set translation data.

        Args:
            transl (Union[np.ndarray, torch.Tensor]):
                Translation in ndarray or tensor,
                in shape [batch_size, 3].

        Raises:
            TypeError: Type of transl is not correct.
        """
        if isinstance(transl, np.ndarray):
            transl_np = transl.reshape(-1, 3)
        elif isinstance(transl, torch.Tensor):
            transl_np = transl.detach().cpu().numpy().reshape(-1, 3)
        else:
            self.logger.error('Type of transl is not correct.\n' +
                              f'Type: {type(transl)}.')
            raise TypeError
        super().__setitem__('transl', transl_np)

    def set_betas(self, betas: Union[np.ndarray, torch.Tensor]) -> None:
        """Set betas data.

        Args:
            betas (Union[np.ndarray, torch.Tensor]):
                Body shape parameters in ndarray or tensor,
                in shape [batch_size, n].
                n stands for any positive int, typically it's 10.

        Raises:
            TypeError: Type of betas is not correct.
        """
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        elif not isinstance(betas, np.ndarray):
            self.logger.error('Type of betas is not correct.\n' +
                              f'Type: {type(betas)}.')
            raise TypeError
        if len(betas.shape) == 1:
            betas = betas[np.newaxis, ...]
        betas_dim = betas.shape[-1]
        betas_np = betas.reshape(-1, betas_dim)
        super().__setitem__('betas', betas_np)

    def __setitem__(self, __k: Any, __v: Any) -> None:
        """Set item according to its key.

        Args:
            __k (Any): Key in dict.
            __v (Any): Value in dict.
        """
        if __k == 'gender':
            self.set_gender(__v)
        elif __k == 'transl':
            self.set_transl(__v)
        elif __k == 'fullpose':
            self.set_fullpose(__v)
        elif __k == 'betas':
            self.set_betas(__v)
        else:
            super().__setitem__(__k, __v)

    def get_batch_size(self) -> int:
        """Get batch size.

        Returns:
            int: batch size of fullpose.
        """
        return self.__getitem__('fullpose').shape[0]

    def get_fullpose(self) -> np.ndarray:
        """Get fullpose.

        Returns:
            ndarray: fullpose in shape [batch_size, fullpose_dim, 3].
        """
        fullpose = self.__getitem__('fullpose')
        return fullpose

    def get_global_orient(self) -> np.ndarray:
        """Get global_orient.

        Returns:
            ndarray: global_orient in shape [batch_size, 3].
        """
        fullpose = self.get_fullpose()
        global_orient = fullpose[:, 0].reshape(-1, 3)
        return global_orient

    def get_body_pose(self) -> np.ndarray:
        """Get body_pose.

        Returns:
            ndarray: body_pose in shape [batch_size, BODY_POSE_LEN, 3].
        """
        fullpose = self.get_fullpose()
        batch_size = self.get_batch_size()
        start_idx = 1
        body_pose = fullpose[:, start_idx:start_idx +
                             self.__class__.BODY_POSE_LEN].reshape(
                                 batch_size, -1, 3)
        return body_pose

    def get_transl(self) -> np.ndarray:
        """Get translation.

        Returns:
            ndarray: translation in shape [batch_size, 3].
        """
        return self.__getitem__('transl').reshape(-1, 3)

    def get_betas(self, repeat_betas: bool = True) -> np.ndarray:
        """Get betas.

        Args:
            repeat_betas (bool, optional):
                Whether to repeat betas when its first dim doesn't match
                batch_size. Defaults to True.

        Returns:
            ndarray:
                betas in shape [batch_size, betas_dims] or
                [1, betas_dims].
        """
        batch_size = self.get_batch_size()
        betas = self.__getitem__('betas')
        if repeat_betas and\
                betas.shape[0] == 1 and\
                betas.shape[0] != batch_size:
            betas = betas.repeat(repeats=batch_size, axis=0)
        return betas

    def to_param_dict(self, repeat_betas: bool = True) -> dict:
        """Split fullpose into global_orient and body_pose, return all the
        necessary parameters in one dict.

        Args:
            repeat_betas (bool, optional):
                Whether to repeat betas when its first dim doesn't match
                batch_size. Defaults to True.

        Returns:
            dict:
                A dict of SMPL data, whose keys are
                betas, body_pose, global_orient and transl.
        """
        body_pose = self.get_body_pose().reshape(self.get_batch_size(), -1)
        global_orient = self.get_global_orient().reshape(
            self.get_batch_size(), 3)
        transl = self.get_transl()
        betas = self.get_betas(repeat_betas=repeat_betas)
        dict_to_return = {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
        }
        return dict_to_return

    def to_tensor_dict(self,
                       repeat_betas: bool = True,
                       device: Union[torch.device, str] = 'cpu') -> dict:
        """It is almost same as self.to_param_dict, but all the values are
        tensors in a specified device. Split fullpose into global_orient and
        body_pose, return all the necessary parameters in one dict.

        Args:
            repeat_betas (bool, optional):
                Whether to repeat betas when its first dim doesn't match
                batch_size. Defaults to True.
            device (Union[torch.device, str], optional):
                A specified device. Defaults to CPU_DEVICE. Defaults to 'cpu'.

        Returns:
            dict: A dict of SMPL data, whose keys are
                betas, body_pose, global_orient and transl.
        """
        np_dict = self.to_param_dict(repeat_betas=repeat_betas)
        dict_to_return = {}
        for key, value in np_dict.items():
            dict_to_return[key] = torch.tensor(
                value, device=device, dtype=torch.float32)
        return dict_to_return

    def dump(self, npz_path: str, overwrite: bool = True):
        """Dump keys and items to an npz file.

        Args:
            npz_path (str):
                Path to a dumped npz file.
            overwrite (bool, optional):
                Whether to overwrite if there is already a file.
                Defaults to True.

        Raises:
            ValueError:
                npz_path does not end with '.npz'.
            FileExistsError:
                When overwrite is False and file exists.
        """
        if not check_path_suffix(npz_path, ['.npz']):
            self.logger.error('Not an npz file.\n' + f'npz_path: {npz_path}')
            raise ValueError
        if not overwrite:
            if check_path_existence(npz_path, 'file') == Existence.FileExist:
                self.logger.error(
                    'File exists while overwrite option not checked.\n' +
                    f'npz_path: {npz_path}')
                raise FileExistsError
        np.savez_compressed(npz_path, **self)

    def from_param_dict(self, smpl_dict: dict) -> None:
        """Load SMPL parameters from smpl_dict.

        Args:
            smpl_dict (dict):
                A dict of ndarray|Tensor parameters.
                global_orient and body_pose are necessary,
                transl and betas are optional.
                Other keys are ignored.


        Raises:
            KeyError: missing necessary keys.
        """
        if not self.__class__.BODY_POSE_KEYS.issubset(smpl_dict):
            self.logger.error('Keys are not enough.\n' +
                              f'smpl_dict\'s keys: {smpl_dict.keys()}')
            raise KeyError
        global_orient = smpl_dict['global_orient']
        batch_size = global_orient.shape[0] \
            if len(global_orient.shape) > 1 else 1
        global_orient = smpl_dict['global_orient'].reshape(batch_size, 3)
        body_pose = smpl_dict['body_pose'].reshape(batch_size, -1)
        if isinstance(global_orient, torch.Tensor):

            def concat_func(data_list, dim):
                return torch.cat(data_list, dim=dim)
        elif isinstance(global_orient, np.ndarray):

            def concat_func(data_list, dim):
                return np.concatenate(data_list, axis=dim)

        fullpose = concat_func([global_orient, body_pose], dim=1)
        self.set_fullpose(fullpose)
        if 'transl' in smpl_dict:
            self.set_transl(smpl_dict['transl'])
        if 'betas' in smpl_dict:
            self.set_betas(smpl_dict['betas'])

    def load(self, npz_path: str):
        """Load data from npz_path and update them to self.

        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        with np.load(npz_path, allow_pickle=True) as npz_file:
            tmp_data_dict = dict(npz_file)
            for key, value in tmp_data_dict.items():
                self.__setitem__(key, value)
