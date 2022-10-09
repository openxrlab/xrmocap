# yapf: disable
import numpy as np
from typing import List, Tuple, Union, overload

from xrmocap.data_structure.keypoints import Keypoints

try:
    from mmhuman3d.core.visualization.visualize_keypoints2d import (
        visualize_kp2d,
    )
    has_mmhuman3d = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmhuman3d = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable


@overload
def visualize_keypoints2d(
        keypoints: Keypoints,
        output_path: str,
        img_arr: np.ndarray,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_keypoints2d(
        keypoints: Keypoints,
        output_path: str,
        img_paths: List[str],
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_keypoints2d(
        keypoints: Keypoints,
        output_path: str,
        video_path: str,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_keypoints2d(
        keypoints: Keypoints,
        output_path: str,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


def visualize_keypoints2d(
        keypoints: Keypoints,
        output_path: str,
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[str]] = None,
        video_path: Union[None, str] = None,
        overwrite: bool = False,
        resolution: Tuple = None,
        return_array: bool = False) -> Union[None, np.ndarray]:
    """Visualize 2d keypoints, powered by mmhuman3d.

    Args:
        keypoints (Keypoints):
            An keypoints2d instance of Keypoints.
        output_path (str):
            Path to the output file. Either a video path
            or a path to an image folder.
        img_arr (Union[None, np.ndarray], optional):
            A single-view image array, in shape
            [n_frame, h, w, c]. Defaults to None.
        img_paths (Union[None, List[str]], optional):
            A list of image paths. Defaults to None.
        video_path (Union[None, str], optional):
            Path to a video file. Defaults to None.
        overwrite (bool, optional):
            Whether replace the file at output_path. Defaults to False.
        return_array (bool, optional):
            Whether to return the visualized image array.
            Defaults to False.

    Raises:
        ValueError: Redundant input.

    Returns:
        Union[None, np.ndarray]:
            If return_array it returns an array of images,
            else return None.
    """
    logger = keypoints.logger
    if not has_mmhuman3d:
        logger.error(import_exception)
        raise ImportError
    # prepare keypoints data
    keypoints_np = keypoints.to_numpy()
    kps2d = keypoints_np.get_keypoints()[..., :2]
    if keypoints_np.get_person_number() == 1:
        kps2d = np.squeeze(kps2d, axis=1)
    kps_convention = keypoints_np.get_convention()
    kps_mask = keypoints_np.get_mask()
    mm_kps_mask = np.sign(np.sum(np.abs(kps_mask), axis=(0, 1)))
    # prepare background data
    input_list = [img_arr, img_paths, video_path]
    input_count = 0
    for input_instance in input_list:
        if input_instance is not None:
            input_count += 1
    if input_count > 1:
        logger.error('Redundant input!\n' + 'Please offer only one among' +
                     ' img_arr, img_paths and video_path.')
        raise ValueError
    vis_arr = visualize_kp2d(
        kp2d=kps2d,
        output_path=output_path,
        image_array=img_arr,
        frame_list=img_paths,
        origin_frames=video_path,
        data_source=kps_convention,
        resolution=resolution,
        mask=mm_kps_mask,
        overwrite=overwrite,
        return_array=return_array)
    return vis_arr
