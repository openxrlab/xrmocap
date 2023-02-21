import numpy as np
from typing import Tuple, Union
from xrprimer.utils.log_utils import get_logger, logging

from .smpl_data import SMPLData
from .smplx_data import SMPLXData
from .smplxd_data import SMPLXDData

__all__ = ['SMPLData', 'SMPLXData', 'SMPLXDData']

_SMPL_DATA_CLASS_DICT = dict(
    SMPLData=SMPLData, SMPLXData=SMPLXData, SMPLXDData=SMPLXDData)


def auto_load_smpl_data(
    npz_path: str,
    logger: Union[None, str, logging.Logger] = None
) -> Tuple[Union[SMPLData, SMPLXData, SMPLXDData], str]:
    """Check which smpl data type the npz file is, and use the correct class to
    load it. Useful when you forget file type.

    Args:
        npz_path (str):
            Path to a dumped npz file.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Returns:
        Union[SMPLData, SMPLXData, SMPLXDData]:
            Loaded SMPL/SMPLX/SMPLXD Data instance.
        str:
            Type(class name) of this npz file.
    """
    logger = get_logger(logger)
    unpacked_dict = dict()
    with np.load(npz_path, allow_pickle=True) as npz_file:
        tmp_data_dict = dict(npz_file)
        for key, value in tmp_data_dict.items():
            if isinstance(value, np.ndarray) and\
                    len(value.shape) == 0:
                # value is not an ndarray before dump
                value = value.item()
            unpacked_dict.__setitem__(key, value)
    if 'fullpose' in unpacked_dict:
        fullpose_dim = unpacked_dict['fullpose'].shape[1]
    else:
        fullpose_dim = 0
    if 'displacement' in unpacked_dict and \
            fullpose_dim == SMPLXDData.get_fullpose_dim():
        type_str = 'SMPLXDData'
    elif 'expression' in unpacked_dict and \
            fullpose_dim == SMPLXData.get_fullpose_dim():
        type_str = 'SMPLXData'
    elif fullpose_dim == SMPLData.get_fullpose_dim():
        type_str = 'SMPLData'
    else:
        logger.error(f'File at {npz_path} is not dumped' +
                     f' by any of {list(_SMPL_DATA_CLASS_DICT.keys())}.')
        raise TypeError
    smpl_data_class = _SMPL_DATA_CLASS_DICT[type_str]
    smpl_data_instance = smpl_data_class.from_dict(unpacked_dict)
    return smpl_data_instance, type_str
