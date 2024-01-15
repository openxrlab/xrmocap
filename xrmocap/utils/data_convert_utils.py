# yapf: disable
import logging
import numpy as np
from enum import Enum
from mmhuman3d.data.data_structures.human_data import HumanData
from typing import Optional, Union

from xrmocap.data_structure.body_model import SMPLData, SMPLXData

# yapf: enable


class SMPLDataTypeEnum(str, Enum):
    SMPLDATA = 'smpl data'
    HUMANDATA = 'human data'
    AMASS = 'AMASS'
    UNKNOWN = 'unknown'


def validate_shape(actual_shape: tuple, expected_shape: tuple) -> bool:
    """Compares the shape of two ndarray.

    Args:
        actual_shape (tuple): the actual shape.
        expected_shape (tuple): the expected shape.

    Returns:
        bool: returns true if the actual shape is the expected shape.
    """
    return all(a == e or e is None
               for a, e in zip(actual_shape, expected_shape))


def validate_spec(specs: dict, data: dict) -> bool:
    """Validate whether the input data conform to the specs.

    Args:
        specs (dict): rules that should be followed.
        data (dict): data to be evaluated.

    Returns:
        bool: returns true if the data follows the specs.
    """
    missing_keys = set(specs.keys()) - set(data.keys())
    if missing_keys:
        return False

    for key, expected_shape in specs.items():
        item = data[key]
        if not validate_shape(item.shape, expected_shape):
            return False
    return True


class SMPLDataConverter:
    """A class that converts the input data into the smpl data."""
    SMPL_DATA_SPECS = {
        'betas': (1, 10),
        'fullpose': (None, 24, 3),
        'gender': (),
        'mask': (None, ),
        'transl': (None, 3)
    }

    SMPLX_DATA_SPECS = {
        'betas': (1, 10),
        'expression': (1, 10),
        'fullpose': (None, 55, 3),
        'gender': (),
        'mask': (None, ),
        'transl': (None, 3)
    }

    AMASS_SMPL_SPECS = {
        'betas': (16, ),
        'gender': (),
        'poses': (None, 156),
        'trans': (None, 3)
    }

    AMASS_SMPLX_SPECS = {
        'betas': (16, ),
        'gender': (),
        'poses': (None, 165),
        'trans': (None, 3)
    }

    HUMANDATA_SMPL_SPECS = {'meta': (), 'smpl': ()}

    HUMANDATA_SMPLX_SPECS = {'meta': (), 'smplx': ()}

    def __init__(self,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """
        Args:
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be
                selected. Defaults to None.
        """
        self.logger = logger

    def get_data_type(self, filepath: str) -> str:
        """Evaluate the data type and the structure of the motion file.

        Args:
            filepath (str): file to evaluate.

        Returns:
            str: the recognized data type.
        """
        try:
            with np.load(filepath, allow_pickle=True) as npz_file:
                data_dict = dict(npz_file)
                if (validate_spec(self.SMPL_DATA_SPECS, data_dict)
                        or validate_spec(self.SMPLX_DATA_SPECS, data_dict)):
                    return SMPLDataTypeEnum.SMPLDATA
                elif (validate_spec(self.AMASS_SMPL_SPECS, data_dict)
                      or validate_spec(self.AMASS_SMPLX_SPECS, data_dict)):
                    return SMPLDataTypeEnum.AMASS
                elif (validate_spec(self.HUMANDATA_SMPL_SPECS, data_dict)
                      or validate_spec(self.HUMANDATA_SMPLX_SPECS, data_dict)):
                    return SMPLDataTypeEnum.HUMANDATA
        except Exception as e:
            self.logger.error({e})

        return SMPLDataTypeEnum.UNKNOWN

    def from_humandata(self,
                       filepath: str) -> Optional[Union[SMPLData, SMPLXData]]:
        """Convert the humandata into the smpl data.

        Args:
            filepath (str): path to the humandata.

        Returns:
            Optional[Union[SMPLData, SMPLXData]]: the resulting smpl data
        """
        human_data = HumanData.fromfile(filepath)
        gender = human_data['meta'].get('gender', None)
        if gender is None:
            gender = 'neutral'
            self.logger.warning(
                f'Cannot find gender record in {human_data}.meta, ' +
                'Use neutral as default.')
        body_model = None
        if 'smpl' in dict(human_data).keys():
            body_model = 'smpl'
        elif 'smplx' in dict(human_data).keys():
            body_model = 'smplx'
        else:
            self.logger.error(
                f'Cannot find body model in {human_data}.meta, ' +
                'supported body models: [smpl, smplx].')
            return None

        betas = human_data[body_model]['betas']
        transl = human_data[body_model]['transl']
        body_pose = human_data[body_model]['body_pose']
        global_orient = human_data[body_model]['global_orient']
        n_frames = body_pose.shape[0]
        mask = np.ones((n_frames, ), dtype=np.uint8)

        res = None
        if 'smpl' == body_model:
            param_dict = dict(
                betas=betas,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose)
            res = SMPLData(gender=gender, logger=self.logger)
            res.from_param_dict(param_dict)
            res.set_mask(mask)
        else:
            param_dict = dict(
                betas=betas,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=human_data['smplx']['left_hand_pose'],
                right_hand_pose=human_data['smplx']['right_hand_pose'],
                leye_pose=human_data['smplx']['leye_pose'],
                reye_pose=human_data['smplx']['reye_pose'],
                jaw_pose=human_data['smplx']['jaw_pose'],
                expression=human_data['smplx']['expression'],
            )
            res = SMPLXData(gender=gender, logger=self.logger)
            res.from_param_dict(param_dict)
            res.set_mask(mask)

        return res

    def from_amass(self,
                   filepath: str) -> Optional[Union[SMPLData, SMPLXData]]:
        """Convert the amass data into the smpl data.

        Args:
            filepath (str): path to the amass data.

        Returns:
            Optional[Union[SMPLData, SMPLXData]]: the resulting smpl data.
        """
        amass_data = np.load(filepath, allow_pickle=True)
        poses = amass_data['poses']
        gender = amass_data['gender']
        betas = amass_data['betas'][:10]
        transl = amass_data['trans']
        global_orient = amass_data['poses'][:, :3]

        n_frames = poses.shape[0]
        mask = np.ones((n_frames, ), dtype=np.uint8)
        res = None
        if poses.shape[1] == 156:  # smpl
            body_pose = amass_data['poses'][:, 3:72]
            param_dict = dict(
                betas=betas,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose)

            res = SMPLData(gender=gender, logger=self.logger)
            res.from_param_dict(param_dict)
            mask = np.ones((n_frames, ), dtype=np.uint8)
            res.set_mask(mask)

        elif poses.shape[1] == 165:  # smplx
            body_pose = amass_data['poses'][:, 3:66]
            jaw_pose = amass_data['poses'][:, 66:69]
            leye_pose = amass_data['poses'][:, 69:72]
            reye_pose = amass_data['poses'][:, 72:75]
            left_hand_pose = amass_data['poses'][:, 75:120]
            right_hand_pose = amass_data['poses'][:, 120:165]
            param_dict = dict(
                betas=betas,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose)
            res = SMPLXData(gender=gender, logger=self.logger)
            res.from_param_dict(param_dict)
            res.set_mask(mask)
        else:
            self.logger.error('Unsupported AMASS data.')

        return res
