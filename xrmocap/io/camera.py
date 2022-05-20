import json
import logging
from typing import Tuple, Union

import numpy as np
from xrprimer.data_structure.camera.pinhole_camera import \
    PinholeCameraParameter  # noqa:E501

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.utils.log_utils import get_logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def load_camera_parameters_from_zoemotion_dir(
        camera_parameter_path: str,
        enable_camera_id: Union[list, None] = None) -> Tuple[list, list]:
    """Load camera parameter and get an RGB PinholeCameraParameter.

    Args:
        camera_parameter_path (str): path to the camera parameter
        enable_camera_id (Union[list, None], optional): camera ID(str).
        Defaults to None.

    Returns:
        cam_param_list (list): PinholeCameraParameter
        enable_camera_list (list): enable camera list e.g.['0', '1']
    """
    enable_camera_list = []
    camera_param_dict = json.load(open(camera_parameter_path))
    if enable_camera_id is None:
        enable_camera_list = [str(x) for x in range(len(camera_param_dict))]
    else:
        enable_camera_list = enable_camera_id.split('_')

    cam_param_list = []
    for camera_id in enable_camera_list:
        cam_param = PinholeCameraParameter()
        cam_param.name = camera_id
        cam_param.k1 = camera_param_dict[camera_id]['distCoeff'][0]
        cam_param.k2 = camera_param_dict[camera_id]['distCoeff'][1]
        cam_param.k3 = camera_param_dict[camera_id]['distCoeff'][4]
        cam_param.p1 = camera_param_dict[camera_id]['distCoeff'][2]
        cam_param.p2 = camera_param_dict[camera_id]['distCoeff'][3]
        cam_param.set_KRT(
            K=camera_param_dict[camera_id]['K'],
            R=np.array(camera_param_dict[camera_id]['R']).reshape(3,
                                                                  3).tolist(),
            T=camera_param_dict[camera_id]['T'],
            world2cam=False)
        cam_param.set_resolution(
            height=camera_param_dict[camera_id]['imgSize'][1],
            width=camera_param_dict[camera_id]['imgSize'][0])
        cam_param_list.append(cam_param)

    return cam_param_list, enable_camera_list


def get_color_camera_parameter_from_smc(
        smc_reader: SMCReader,
        camera_type: Literal['kinect', 'iphone'],
        camera_id: int,
        logger: Union[None, str,
                      logging.Logger] = None) -> PinholeCameraParameter:
    """Get an RGB PinholeCameraParameter from an smc reader.

    Args:
        smc_reader (SMCReader):
            An SmcReader instance containing kinect
            and iphone camera parameters.
        camera_type (Literal['kinect', 'iphone']):
            Which type of camera to get.
        camera_id (int):
            ID of the selected camera.
        logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

    Raises:
        NotImplementedError: iphone has not been supported yet.
        KeyError: camera_type is neither kinect nor iphone.

    Returns:
        PinholeCameraParameter
    """
    logger = get_logger(logger)
    cam_param = PinholeCameraParameter(name=f'{camera_type}_{camera_id:02d}')
    if camera_type == 'kinect':
        extrinsics_dict = \
            smc_reader.get_kinect_color_extrinsics(
                camera_id, homogeneous=False
            )
        extrinsics_r_np = extrinsics_dict['R'].reshape(3, 3)
        extrinsics_t_np = extrinsics_dict['T'].reshape(3)
        intrinsics_np = \
            smc_reader.get_kinect_color_intrinsics(
                camera_id
            )
        resolution = \
            smc_reader.get_kinect_color_resolution(
                camera_id
            )
        cam_param.set_KRT(
            K=intrinsics_np.tolist(),
            R=extrinsics_r_np.tolist(),
            T=extrinsics_t_np.tolist(),
            world2cam=False)
        cam_param.set_resolution(
            height=int(resolution[1]), width=int(resolution[0]))
    elif camera_type == 'iphone':
        raise NotImplementedError('iphone has not been supported yet.')
    else:
        logger.error('Choose camera_type from [\'kinect\', \'iphone\'].')
        raise KeyError('Wrong camera_type.')
    return cam_param
