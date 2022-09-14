# yapf: disable
import numpy as np
from typing import List, Union, overload
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.opencv_projector import OpencvProjector

from xrmocap.data_structure.keypoints import Keypoints
from .visualize_keypoints2d import visualize_keypoints2d

try:
    from mmhuman3d.core.visualization.visualize_keypoints3d import (
        visualize_kp3d,
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
def visualize_project_keypoints3d(
        keypoints: Keypoints,
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        output_path: str,
        img_arr: np.ndarray,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_project_keypoints3d(
        keypoints: Keypoints,
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        output_path: str,
        img_paths: List[str],
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_project_keypoints3d(
        keypoints: Keypoints,
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        output_path: str,
        video_path: str,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


@overload
def visualize_project_keypoints3d(
        keypoints: Keypoints,
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        output_path: str,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    ...


def visualize_project_keypoints3d(
        keypoints: Keypoints,
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        output_path: str,
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[str]] = None,
        video_path: Union[None, str] = None,
        overwrite: bool = False,
        return_array: bool = False) -> Union[None, np.ndarray]:
    """Project 3d keypoints to 2d, visualize the peojected keypoints, powered
    by mmhuman3d.

    Args:
        keypoints (Keypoints):
            An keypoints3d instance of Keypoints.
        cam_param (Union[FisheyeCameraParameter, PinholeCameraParameter]):
            Camera parameter, either an instance of FisheyeCameraParameter
            or PinholeCameraParameter.
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
    projector = OpencvProjector(camera_parameters=[cam_param])
    projected_kps2d = projector.project(
        points=keypoints.get_keypoints()[..., :3].reshape(-1, 3),
        points_mask=np.expand_dims(keypoints.get_mask(), axis=-1))
    projected_kps2d = projected_kps2d.reshape(keypoints.get_frame_number(),
                                              keypoints.get_person_number(),
                                              keypoints.get_keypoints_number(),
                                              2)
    projected_kps2d = np.concatenate(
        (projected_kps2d, np.ones_like(projected_kps2d[..., 0:1])), axis=-1)
    keypoints_np.set_keypoints(projected_kps2d)
    vis_arr = visualize_keypoints2d(
        keypoints=keypoints_np,
        output_path=output_path,
        img_arr=img_arr,
        img_paths=img_paths,
        video_path=video_path,
        overwrite=overwrite,
        return_array=return_array)
    return vis_arr


def visualize_keypoints3d(
        keypoints: Keypoints,
        output_path: str,
        return_array: bool = False) -> Union[None, np.ndarray]:
    """Visualize 3d keypoints, powered by mmhuman3d.

    Args:
        keypoints (Keypoints):
            An keypoints3d instance of Keypoints.
        output_path (str):
            Path to the output file. Either a video path
            or a path to an image folder.
        return_array (bool, optional):
            Whether to return the visualized image array.
            Defaults to False.

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
    kps3d = keypoints_np.get_keypoints()[..., :3]
    if keypoints_np.get_person_number() == 1:
        kps3d = np.squeeze(kps3d, axis=1)
    kps_convention = keypoints_np.get_convention()
    kps_mask = keypoints_np.get_mask()
    mm_kps_mask = np.sign(np.sum(np.abs(kps_mask), axis=(0, 1)))
    vis_arr = visualize_kp3d(
        kp3d=kps3d,
        output_path=output_path,
        data_source=kps_convention,
        mask=mm_kps_mask,
        return_array=return_array)
    return vis_arr
