# yapf: disable
import numpy as np
from typing import List, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.data_structure.keypoints import Keypoints
from xrprimer.ops.projection.builder import OpencvProjector
from xrprimer.utils.log_utils import get_logger, logging

from .visualize_keypoints2d import visualize_keypoints2d

# yapf: enable


def visualize_keypoints3d_projected(
    # input args
    keypoints: Keypoints,
    camera: Union[PinholeCameraParameter, FisheyeCameraParameter],
    # output args
    output_path: str,
    overwrite: bool = True,
    return_array: bool = False,
    plot_points: bool = True,
    plot_lines: bool = True,
    # background args
    background_arr: Union[np.ndarray, None] = None,
    background_dir: Union[np.ndarray, None] = None,
    background_video: Union[np.ndarray, None] = None,
    background_img_list: Union[List[str], None] = None,
    height: Union[int, None] = None,
    width: Union[int, None] = None,
    # verbose args
    disable_tqdm: bool = True,
    logger: Union[None, str,
                  logging.Logger] = None) -> Union[None, np.ndarray]:
    """Visualize multi-frame keypoints3d by OpenCV, overlay with 2D images. For
    plot args, please either plot_points or plot_lines, or both. For background
    args, please offer only one of them.

    Args:
        keypoints (Keypoints):
            An instance of class Keypoints. If n_person > 1,
            each person has its own color, else each point
            and line has different color.
        camera (Union[PinholeCameraParameter, FisheyeCameraParameter]):
            Camera parameter of a camera from which we watch the 3D
            keypoints.
        output_path (str):
            Path to the output mp4 video file or image directory.
        overwrite (bool, optional):
            Whether to overwrite the file at output_path.
            Defaults to True.
        return_array (bool, optional):
            Whether to return the video array. If True,
            please make sure your RAM is enough for the video.
            Defaults to False, return None.
        plot_points (bool, optional):
            Whether to plot points according to keypoints'
            location.
            Defaults to True.
        plot_lines (bool, optional):
            Whether to plot lines according to keypoints'
            limbs. Defaults to True.
        background_arr (Union[np.ndarray, None], optional):
            Background image array. Defaults to None.
        background_dir (Union[str, None], optional):
            Path to the image directory for background.
            Defaults to None.
        background_video (Union[str, None], optional):
            Path to the video for background.
            Defaults to None.
        background_img_list (Union[List[str], None], optional):
            List of paths to images for background.
            Defaults to None.
        height (Union[int, None], optional):
            Height of background. Defaults to None.
        width (Union[int, None], optional):
            Width of background. Defaults to None.
        disable_tqdm (bool, optional):
            Whether to disable tqdm progress bar.
            Defaults to True.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Raises:
        ValueError: Neither plot_points nor plot_lines is True.

    Returns:
        Union[np.ndarray, None]:
            Plotted multi-frame image array or None.
            If it's an array, its shape shall be
            [n_frame, height, width, 3].
    """
    logger = get_logger(logger)
    if not plot_points and not plot_lines:
        logger.error('plot_points or plot_lines must be True.')
        raise ValueError
    # construct a projector
    projector = OpencvProjector(camera_parameters=[camera], logger=logger)
    n_frame = keypoints.get_frame_number()
    n_person = keypoints.get_person_number()
    n_kps = keypoints.get_keypoints_number()
    # project 3d keypoints to 2d
    projected_kps2d = projector.project(
        points=keypoints.get_keypoints()[..., :3].reshape(
            n_frame * n_person * n_kps, 3),
        points_mask=keypoints.get_mask().reshape(n_frame * n_person * n_kps,
                                                 1))
    # concat kps2d with kps3d's confidence
    projected_kps2d = projected_kps2d[0].reshape(n_frame, n_person, n_kps, 2)
    projected_kps2d = np.concatenate(
        (projected_kps2d, keypoints.get_keypoints()[..., 3:]), axis=-1)
    keypoints2d = keypoints.clone()
    # mask is not changed, set keypoints data is enough
    keypoints2d.set_keypoints(
        projected_kps2d.reshape(n_frame, n_person, n_kps, 3))
    # call visualize_keypoints2d
    ret_value = visualize_keypoints2d(
        keypoints=keypoints2d,
        output_path=output_path,
        overwrite=overwrite,
        return_array=return_array,
        plot_points=plot_points,
        plot_lines=plot_lines,
        background_arr=background_arr,
        background_dir=background_dir,
        background_video=background_video,
        background_img_list=background_img_list,
        height=height,
        width=width,
        disable_tqdm=disable_tqdm,
        logger=logger)
    return ret_value
