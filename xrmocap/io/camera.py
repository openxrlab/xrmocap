import logging
from typing import Union

from xrprimer.data_structure.camera.pinhole_camera import \
    PinholeCameraParameter  # PinholeCamera with distortion

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.utils.log_utils import get_logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


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
