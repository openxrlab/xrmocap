# yapf: disable
import json
import logging
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import List, Tuple, Union
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.camera.extrinsic import rotate_camera
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.smc_reader import SMCReader

# yapf: enable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def load_camera_parameters_from_zoemotion_dir(
        camera_parameter_path: str,
        enable_camera_id: Union[list, None] = None) -> Tuple[list, list]:
    """Load camera parameter and get an RGB FisheyeCameraParameter.

    Args:
        camera_parameter_path (str): path to the camera parameter
        enable_camera_id (Union[list, None], optional): camera ID(str).
        Defaults to None.

    Returns:
        cam_param_list (list): FisheyeCameraParameter
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
        dist_coeff_k = []
        dist_coeff_p = []
        cam_param = FisheyeCameraParameter()
        cam_param.name = camera_id
        dist_coeff_k = [
            camera_param_dict[camera_id]['distCoeff'][0],
            camera_param_dict[camera_id]['distCoeff'][1],
            camera_param_dict[camera_id]['distCoeff'][4], 0, 0, 0
        ]
        dist_coeff_p = [
            camera_param_dict[camera_id]['distCoeff'][2],
            camera_param_dict[camera_id]['distCoeff'][3]
        ]
        cam_param.set_dist_coeff(dist_coeff_k, dist_coeff_p)
        cam_param.set_KRT(
            K=np.array(camera_param_dict[camera_id]['K']).reshape(3, 3),
            R=np.array(camera_param_dict[camera_id]['R']).reshape(3, 3),
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
        align_floor: bool = True,
        logger: Union[None, str,
                      logging.Logger] = None) -> FisheyeCameraParameter:
    """Get an RGB FisheyeCameraParameter from an smc reader.

    Args:
        smc_reader (SMCReader):
            An SmcReader instance containing kinect
            and iphone camera parameters.
        camera_type (Literal['kinect', 'iphone']):
            Which type of camera to get.
        camera_id (int):
            ID of the selected camera.
        align_floor (bool):
            Whether to rotate camera extrinsics, makes the xOz
            plane parallel to floor calibrated by Kinect_0's
            depth camera.
            Defaults to True.
        logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

    Raises:
        NotImplementedError: iphone has not been supported yet.
        KeyError: camera_type is neither kinect nor iphone.

    Returns:
        FisheyeCameraParameter
    """
    logger = get_logger(logger)
    cam_param = FisheyeCameraParameter(name=f'{camera_type}_{camera_id:02d}')
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
    if align_floor:
        floor_dict = smc_reader.get_depth_floor(0)
        calib_space_normal = np.array(floor_dict['normal']).reshape((3))
        opencv_space_normal = np.asarray([0, -1, 0])
        rotation = scipy_Rotation.align_vectors(
            np.expand_dims(opencv_space_normal, 0),
            np.expand_dims(calib_space_normal, 0))[0]
        cam_rot_mat = rotation.as_matrix()
        cam_param = rotate_camera(cam_param, cam_rot_mat)
    return cam_param


def get_all_color_kinect_parameter_from_smc(
    smc_reader: SMCReader,
    align_floor: bool = True,
    logger: Union[None, str, logging.Logger] = None
) -> List[FisheyeCameraParameter]:
    """Get an RGB FisheyeCameraParameter from an smc reader.

    Args:
        smc_reader (SMCReader):
            An SmcReader instance containing kinect
            and iphone camera parameters.
        align_floor (bool):
            Whether to rotate camera extrinsics, makes the xOz
            plane parallel to floor calibrated by Kinect_0's
            depth camera.
            Defaults to True.
        logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

    Returns:
        List[FisheyeCameraParameter]
    """
    logger = get_logger(logger)
    ret_list = []
    for kinect_index in range(smc_reader.num_kinects):
        cam_param = get_color_camera_parameter_from_smc(
            smc_reader=smc_reader,
            camera_type='kinect',
            camera_id=kinect_index,
            align_floor=False,
            logger=logger)
        ret_list.append(cam_param)
    if align_floor:
        floor_dict = smc_reader.get_depth_floor(0)
        calib_space_normal = np.array(floor_dict['normal']).reshape((3))
        opencv_space_normal = np.asarray([0, -1, 0])
        rotation = scipy_Rotation.align_vectors(
            np.expand_dims(opencv_space_normal, 0),
            np.expand_dims(calib_space_normal, 0))[0]
        cam_rot_mat = rotation.as_matrix()
        for kinect_index, cam_param in enumerate(ret_list):
            ret_list[kinect_index] = rotate_camera(cam_param, cam_rot_mat)
    return ret_list
