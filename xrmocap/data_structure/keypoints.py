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


class Keypoints(dict):
    """A class for multi-frame, multi-person keypoints data, based on python
    dict.

    keypoints, mask and convention are the three necessary keys, and we advise
    you to just call Keypoints(). If you'd like to set them manually, it is
    recommended to obey the following turn: convention -> keypoints -> mask.
    """

    def __init__(self,
                 src_dict: dict = None,
                 dtype: Literal['torch', 'numpy', 'auto'] = 'auto',
                 kps: Union[np.ndarray, torch.Tensor, None] = None,
                 mask: Union[np.ndarray, torch.Tensor, None] = None,
                 convention: Union[str, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a Keypoints instance with pre-set values. If any of kps,
        mask, convention is provided, it will override the item in src_dict.

        Args:
            src_dict (dict, optional):
                A dict with items in Keypoints fashion.
                Defaults to None.
            dtype (Literal['torch', 'numpy', 'auto'], optional):
                The data type of this Keypoints instance, values will
                be converted to the certain dtype when setting. If
                dtype==auto, it be changed the first time set_keypoints()
                is called, and never changes.
                Defaults to 'auto'.
            kps (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for keypoints,
                kps2d in shape [n_frame, n_person, n_kps, 3],
                kps3d in shape [n_frame, n_person, n_kps, 4].
                Shape [n_kps, 3 or 4] is also accepted, unsqueezed
                automatically. Defaults to None.
            mask (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for keypoint mask,
                in shape [n_frame, n_person, n_kps],
                in dtype uint8.
                Shape [n_kps, ] is also accepted, unsqueezed
                automatically. Defaults to None.
            convention (str, optional):
                Convention name of the keypoints,
                can be found in KEYPOINTS_FACTORY.
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        if src_dict is not None:
            super().__init__(src_dict)
        else:
            super().__init__()
        self.logger = get_logger(logger)

        if dtype == 'auto':
            if kps is not None:
                dtype = __get_array_type_str__(kps, logger)
            elif src_dict is not None and 'keypoints' in src_dict:
                dtype = __get_array_type_str__(src_dict['keypoints'], logger)
        self.dtype = dtype

        if convention is not None:
            self.set_convention(convention)

        if kps is not None:
            self.set_keypoints(kps)

        if mask is None and 'mask' not in self and\
                'keypoints' in self:
            default_n_kps = self.get_keypoints_number()
            mask = np.ones(shape=(default_n_kps, ))
        if mask is not None:
            self.set_mask(mask)

    @classmethod
    def fromfile(cls, npz_path: str) -> 'Keypoints':
        """Construct a body model data structure from an npz file.

        Args:
            npz_path (str):
                Path to a dumped npz file.

        Returns:
            Keypoints:
                A Keypoints instance load from file.
        """
        ret_instance = cls()
        ret_instance.load(npz_path)
        return ret_instance

    def set_keypoints(self, kps: Union[np.ndarray, torch.Tensor]) -> None:
        """Set keypoints array.

        Args:
            kps (Union[np.ndarray, torch.Tensor]):
                A tensor or ndarray for keypoints,
                kps2d in shape [n_frame, n_person, n_kps, 3],
                kps3d in shape [n_frame, n_person, n_kps, 4].
                Shape [n_kps, 3 or 4] is also accepted, unsqueezed
                automatically.

        Raises:
            TypeError: Type of keypoints is wrong.
            ValueError: kps.shape[-1] is wrong.
            ValueError: Shape of kps is wrong.
        """
        if self.dtype == 'auto':
            self.dtype = __get_array_type_str__(kps, self.logger)
        keypoints = __get_array_in_type__(
            array=kps, type=self.dtype, logger=self.logger)
        # shape: frame_n, person_n, kp_n, dim+score
        if keypoints.shape[-1] not in (3, 4):
            self.logger.error('shape[-1] of kps2d should be 3,' +
                              ' shape[-1] of kps3d should be 4.\n' +
                              f'kps.shape[-1]: {kps.shape[-1]}.')
            raise ValueError
        if len(keypoints.shape) == 2:
            keypoints = keypoints.reshape(1, 1, keypoints.shape[0],
                                          keypoints.shape[1])
        if len(keypoints.shape) != 4:
            self.logger.error('Shape of keypoints should be' +
                              ' [n_frame, n_person, n_kps, dim+1].\n' +
                              f'kps.shape: {kps.shape}.')
            raise ValueError
        super().__setitem__('keypoints', keypoints)

    def set_convention(self, convention: str) -> None:
        """Set convention name of the keypoints.

        Args:
            convention (str):
                Convention name of the keypoints,
                can be found in KEYPOINTS_FACTORY.

        Raises:
            TypeError: Type of convention is not str.
        """
        if not isinstance(convention, str):
            self.logger.error('Type of convention is not str.\n' +
                              f'type(convention): {type(convention)}.')
            raise TypeError
        super().__setitem__('convention', convention)

    def set_mask(self, mask: Union[np.ndarray, torch.Tensor]) -> None:
        """Set mask of the keypoints. It should be called after the
        corresponding keypoints has been set.

        Args:
            mask (Union[np.ndarray, torch.Tensor]):
                A tensor or ndarray for keypoint mask,
                in shape [n_frame, n_person, n_kps],
                in dtype uint8.
                Shape [n_kps, ] is also accepted, unsqueezed
                automatically.

        Raises:
            TypeError: Type of mask is wrong.
            ValueError: Shape of mask is wrong.
        """
        if self.dtype == 'auto':
            self.dtype = __get_array_type_str__(mask, self.logger)

        mask = __get_array_in_type__(
            array=mask, type=self.dtype, logger=self.logger)

        if self.dtype == 'torch':

            def to_type_uint8_func(data):
                return data.to(dtype=torch.uint8)

        else:

            def to_type_uint8_func(data):
                return data.astype(np.uint8)

        mask = to_type_uint8_func(mask)
        keypoints_shape = self.get_keypoints().shape
        if len(mask.shape) == 1:
            mask = mask.reshape(1, 1, len(mask))
            mask = mask.repeat(keypoints_shape[0], axis=0)
            mask = mask.repeat(keypoints_shape[1], axis=1)
        if len(mask.shape) != 3 or \
                mask.shape != keypoints_shape[:3]:
            self.logger.error('Shape of mask should be' +
                              ' [n_frame, n_person, n_kps].\n' +
                              f'mask.shape: {mask.shape}.' +
                              f'keypoints.shape: {keypoints_shape}.')
            raise ValueError
        super().__setitem__('mask', mask)

    def __setitem__(self, __k: Any, __v: Any) -> None:
        """Set item according to its key.

        Args:
            __k (Any): Key in dict.
            __v (Any): Value in dict.
        """
        if __k == 'keypoints':
            self.set_keypoints(__v)
        elif __k == 'convention':
            self.set_convention(__v)
        elif __k == 'mask':
            self.set_mask(__v)
        else:
            super().__setitem__(__k, __v)

    def get_keypoints(self) -> Union[np.ndarray, torch.Tensor]:
        """Get keypoints array.

        Returns:
            np.ndarray: keypoints
        """
        return self['keypoints']

    def get_mask(self) -> Union[np.ndarray, torch.Tensor]:
        """Get keypoints mask.

        Returns:
            np.ndarray: mask
        """
        return self['mask']

    def get_convention(self) -> str:
        """Get keypoints convention name.

        Returns:
            str: convention
        """
        return self['convention']

    def get_frame_number(self) -> int:
        """Get frame number of keypoints.

        Returns:
            int: frame number
        """
        return self.get_keypoints().shape[0]

    def get_person_number(self) -> int:
        """Get person number of keypoints.

        Returns:
            int: person number
        """
        return self.get_keypoints().shape[1]

    def get_keypoints_number(self) -> int:
        """Get number of keypoints.

        Returns:
            int: keypoints number
        """
        return self.get_keypoints().shape[2]

    def to_tensor(self,
                  device: Union[torch.device, str] = 'cpu') -> 'Keypoints':
        """Return all the necessary values for keypoints expression in another
        Keypoints instance, convert ndarray into Tensor.

        Args:
            device (Union[torch.device, str], optional):
                A specified device.
                Defaults to 'cpu'.

        Returns:
            Keypoints: An instance of Keypoints data, whose keys are
                keypoints, mask, convention.
        """
        kps_to_return = self.__class__(
            dtype='torch',
            kps=self.get_keypoints(),
            mask=self.get_mask(),
            convention=self.get_convention(),
            logger=self.logger)
        kps_to_return.set_keypoints(kps_to_return.get_keypoints().to(device))
        kps_to_return.set_mask(kps_to_return.get_mask().to(device))
        return kps_to_return

    def to_numpy(self, ) -> 'Keypoints':
        """Return all the necessary values for keypoints expression in another
        Keypoints instance, convert Tensor into numpy.

        Returns:
            Keypoints: An instance of Keypoints data, whose keys are
                keypoints, mask, convention.
        """
        kps_to_return = self.__class__(
            dtype='numpy',
            kps=self.get_keypoints(),
            mask=self.get_mask(),
            convention=self.get_convention(),
            logger=self.logger)
        return kps_to_return

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
        if self.dtype == 'numpy':
            dict_to_save = self
        else:  # else self.dtype == tensor
            dict_to_save = self.to_numpy()
        np.savez_compressed(npz_path, **dict_to_save)

    def load(self, npz_path: str):
        """Load data from npz_path and update them to self.

        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        with np.load(npz_path, allow_pickle=True) as npz_file:
            tmp_data_dict = dict(npz_file)
            for key, value in tmp_data_dict.items():
                if isinstance(value, np.ndarray) and\
                        len(value.shape) == 0:
                    # value is not an ndarray before dump
                    value = value.item()
                self.__setitem__(key, value)

    def clone(self) -> 'Keypoints':
        """Clone a Keypoints instance as self.

        Returns:
            Keypoints:
                A deep copied instance of Keypoints,
                with the same dtype and value as self.
        """
        ret_kps = self.__class__(
            dtype=self.dtype,
            kps=__copy_array_tensor__(self.get_keypoints()),
            mask=__copy_array_tensor__(self.get_mask()),
            convention=self.get_convention(),
            logger=self.logger)
        return ret_kps


def __get_array_type_str__(array, logger) -> Literal['torch', 'numpy']:
    if isinstance(array, torch.Tensor):
        return 'torch'
    elif isinstance(array, np.ndarray):
        return 'numpy'
    else:
        logger = get_logger(logger)
        logger.error('Type of array is not correct.\n' +
                     f'Type: {type(array)}.')
        raise TypeError


def __get_array_in_type__(array: Union[torch.Tensor, np.ndarray],
                          type: Literal['torch', 'numpy'],
                          logger: Union[None, str, logging.Logger]):
    logger = get_logger(logger)
    if type == 'numpy':
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        elif not isinstance(array, np.ndarray):
            logger.error('Type of array is not correct.\n' +
                         f'Type: {type(array)}.')
    else:  # type == 'torch'
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        elif not isinstance(array, torch.Tensor):
            logger.error('Type of array is not correct.\n' +
                         f'Type: {type(array)}.')
    return array


def __copy_array_tensor__(
        data: Union[np.ndarray,
                    torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, np.ndarray):
        return data.copy()
    else:
        return data.clone()
