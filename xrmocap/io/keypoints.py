import logging
import numpy as np
import os
import pickle as pkl
from typing import Union

from xrmocap.utils.log_utils import get_logger


def load_keypoints2d_from_zoemotion_pkl(kps2d_data_dir: str,
                                        enable_camera_list: list,
                                        logger: Union[None, str,
                                                      logging.Logger] = None):
    """Load keypoints2d data of the selected views in enable_camera_list from
    zoehuman, including 'id', 'mask', 'keypoints', 'bbox' and 'mask'.

    Args:
        kps2d_data_dir (str): The path to the keypoints2d data from zoemotion.
        enable_camera_list (list): Selected camera ID.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Returns:
        mmpose_result_list (list[dict]): The result from mmpose.
        mask (numpy.ndarray): The mask for keypoints validation.
    """
    mmpose_path_dict = {}
    mmpose_path_list = []
    mmpose_result_list = []
    logger = get_logger(logger)

    for camera_id in enable_camera_list:
        mmpose_path_dict[camera_id] = \
            os.path.join(kps2d_data_dir, f'cam{camera_id}_2dkeypoint.pickle')
        mmpose_path_list.append(mmpose_path_dict[camera_id])

    camera_number = len(enable_camera_list)
    for i in range(camera_number):
        with open(mmpose_path_list[i], 'rb') as f_readb:
            kps2d_data = pkl.load(f_readb, encoding='bytes')
            kps2d_data_dict = {}
            mask = None
            keys = [
                'keypoints', 'bbox', 'id', 'heatmap_bbox', 'heatmap_data',
                'cropped_img'
            ]
            for raw_image_name in kps2d_data.keys():
                if len(
                        kps2d_data[raw_image_name]
                ) > 0 and 'keypoints' in kps2d_data[raw_image_name][0].keys():
                    kps2d_data_dict[raw_image_name] = {}
                    for key in keys:
                        if key in kps2d_data[raw_image_name][0]:
                            kps2d_data_dict[raw_image_name][key] = [
                                kps2d_data[raw_image_name][i][key]
                                for i in range(
                                    len(kps2d_data[raw_image_name]))
                            ]
                    mask = kps2d_data[raw_image_name][0]['mask']
                else:
                    kps2d_data_dict[raw_image_name] = None

        mmpose_result_list.append(kps2d_data_dict)

        if 'panoptic' in kps2d_data_dir and mask is None:
            mask = np.full(17, True)

        if mask is None:
            logger.info(f'cam {i} detect no human')

    assert mask is not None

    return mmpose_result_list, mask
