import cv2
import logging
import numpy as np
from typing import List, Union
from xrprimer.utils.ffmpeg_utils import VideoInfoReader, video_to_array


def load_multiview_images(
        img_paths: Union[None, List[List[str]]]) -> np.ndarray:
    """Load multi-view images to an ndarray.

    Args:
        img_paths (Union[None, List[List[str]]]):
            A nested list of image paths, in shape
            [view_n, frame_n].

    Returns:
        np.ndarray:
            Multi-view image array, in shape
            [view_n, frame_n, h, w, c].
    """
    # multi-view list
    mview_list = []
    for view_list in img_paths:
        # single-view list
        sv_list = []
        for img_path in view_list:
            frame_np = cv2.imread(filename=img_path)
            sv_list.append(frame_np)
        mview_list.append(sv_list)
    mview_array = np.asarray(mview_list)
    return mview_array


def get_n_frame_from_mview_src(
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[List[str]]] = None,
        video_paths: Union[None, List[str]] = None,
        logger: Union[None, str, logging.Logger] = None) -> int:
    """Get number of frames from multi-view image source. It needs one images
    input among [img_arr, img_paths, video_paths].

    Args:
        img_arr (Union[None, np.ndarray], optional):
            A multi-view image array, in shape
            [n_view, n_frame, h, w, c]. Defaults to None.
        img_paths (Union[None, List[List[str]]], optional):
            A nested list of image paths, in shape
            [n_view, n_frame]. Defaults to None.
        video_paths (Union[None, List[str]], optional):
            A list of video paths, each is a view.
            Defaults to None.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Returns:
        int: Number of frames.
    """
    n_frame = None
    if img_arr is not None:
        n_frame = img_arr.shape[1]
    elif img_paths is not None:
        n_frame = len(img_paths[0])
    elif video_paths is not None:
        reader = VideoInfoReader(video_paths[0], logger=logger)
        n_frame = int(reader['nb_frames'])
    return n_frame


def load_clip_from_mview_src(
        start_idx: int,
        end_idx: int,
        img_arr: Union[None, np.ndarray] = None,
        img_paths: Union[None, List[List[str]]] = None,
        video_paths: Union[None, List[str]] = None,
        logger: Union[None, str, logging.Logger] = None) -> np.ndarray:
    """Get image array of a clip from multi-view image source. It needs one
    images input among [img_arr, img_paths, video_paths].

    Args:
        img_arr (Union[None, np.ndarray], optional):
            A multi-view image array, in shape
            [n_view, n_frame, h, w, c]. Defaults to None.
        img_paths (Union[None, List[List[str]]], optional):
            A nested list of image paths, in shape
            [n_view, n_frame]. Defaults to None.
        video_paths (Union[None, List[str]], optional):
            A list of video paths, each is a view.
            Defaults to None.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Returns:
        np.ndarray: Image array of the selected range.
    """
    ret_img_arr = None
    if img_arr is not None:
        ret_img_arr = img_arr[:, start_idx:end_idx, ...]
    elif img_paths is not None:
        ret_img_list = []
        for view_idx in range(len(img_paths)):
            view_img_list = []
            for frame_idx in range(start_idx, end_idx):
                img = cv2.imread(img_paths[view_idx][frame_idx])
                view_img_list.append(img)
            ret_img_list.append(view_img_list)
        ret_img_arr = np.asarray(ret_img_list)
    elif video_paths is not None:
        ret_img_list = []
        for view_idx in range(len(video_paths)):
            img_arr = video_to_array(
                video_paths[view_idx],
                start=start_idx,
                end=end_idx,
                logger=logger)
            ret_img_list.append(img_arr)
        ret_img_arr = np.asarray(ret_img_list)
    return ret_img_arr
